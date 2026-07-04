#!/usr/bin/env python3
"""
automl_zero.py — Evolucion de ALGORITMOS DE APRENDIZAJE desde cero (AutoML-Zero),
acelerado por GPU, meta-entrenado sobre una distribucion de tareas.

Idea (Real et al., ICML 2020, adaptado a full-batch + GPU):
  Cada individuo es un ALGORITMO con tres funciones sobre una memoria de registros
  tipados (escalares sX, vectores vX de dim F, y buffers por-ejemplo pX de dim N):
      Setup()      -> inicializa pesos / hiperparametros
      Predict(X)   -> produce y_hat (queda en p0)                 [NO puede ver y]
      Learn(X, y)  -> ACTUALIZA los pesos dado el error           [la regla ADAPTATIVA]
  Se entrena desde cero via el Learn EVOLUCIONADO y se evalua en validacion.
  El fitness es el R2 medio sobre VARIAS tareas -> premia algoritmos que GENERALIZAN
  (meta-learning: "aprender a aprender", no memorizar una tarea).

Reusa el espiritu del motor GP (elitismo, restart por estancamiento, hall-of-fame),
pero el genoma son 3 secuencias de instrucciones y el fitness entrena-y-evalua.

Verificacion: `--verify` corre un algoritmo de regresion lineal por gradiente
CODIFICADO A MANO y comprueba que el VM entrena (R2 alto) antes de evolucionar.
"""
import argparse
import time
import numpy as np
import torch

# ----------------------------------------------------------------------------
# 1. MEMORIA Y CONJUNTO DE OPERACIONES (VM de registros tipados)
#    Tipos: 'S' escalar [], 'V' vector [F], 'P' por-ejemplo [N].
# ----------------------------------------------------------------------------
# op: (nombre, tipo_out, tipo_in1, tipo_in2|None, usa_imm)
OPS = [
    ("S_ADD", "S", "S", "S", False), ("S_SUB", "S", "S", "S", False),
    ("S_MUL", "S", "S", "S", False), ("S_DIV", "S", "S", "S", False),
    ("S_CONST", "S", None, None, True), ("S_MEANP", "S", "P", None, False),
    ("S_DOT", "S", "V", "V", False),
    ("V_ADD", "V", "V", "V", False), ("V_SUB", "V", "V", "V", False),
    ("V_MUL", "V", "V", "V", False), ("V_SCALE", "V", "V", "S", False),
    ("V_RELU", "V", "V", None, False),
    ("V_MATVEC_XT", "V", "P", None, False),          # v = X^T @ p / N   (direccion tipo gradiente)
    ("P_MATVEC_X", "P", "V", None, False),           # p = X @ v         (prediccion lineal)
    ("P_ADD", "P", "P", "P", False), ("P_SUB", "P", "P", "P", False),
    ("P_MUL", "P", "P", "P", False), ("P_SCALE", "P", "P", "S", False),
    ("P_ADDS", "P", "P", "S", False), ("P_RELU", "P", "P", None, False),
    ("P_TANH", "P", "P", None, False),
    ("P_LOADY", "P", None, None, False),             # p = y   (SOLO permitido en Learn/Setup)
]
OP = {o[0]: o for o in OPS}
IMM_POOL = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, -1.0, -0.5, -0.1]

# op-sets por funcion (Predict NO puede leer y -> sin P_LOADY, evita fuga de etiqueta)
OPS_SETUP = [o[0] for o in OPS if o[0] != "P_LOADY"]
OPS_PREDICT = [o[0] for o in OPS if o[0] != "P_LOADY"]
OPS_LEARN = [o[0] for o in OPS]


class Mem:
    def __init__(self, nS, nV, nP, F, N, device):
        self.S = torch.zeros(nS, device=device)
        self.V = torch.zeros(nV, F, device=device)
        self.P = torch.zeros(nP, N, device=device)


def _fix(t):
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def run(instrs, mem, X, y, nS, nV, nP):
    """Ejecuta una lista de instrucciones mutando mem. X:[N,F], y:[N]."""
    N = X.shape[0]
    for (name, o, a, b, imm) in instrs:
        t = OP[name]
        if name == "S_ADD": mem.S[o] = mem.S[a] + mem.S[b]
        elif name == "S_SUB": mem.S[o] = mem.S[a] - mem.S[b]
        elif name == "S_MUL": mem.S[o] = mem.S[a] * mem.S[b]
        elif name == "S_DIV": mem.S[o] = mem.S[a] / mem.S[b] if abs(float(mem.S[b])) > 1e-6 else mem.S[a]
        elif name == "S_CONST": mem.S[o] = imm
        elif name == "S_MEANP": mem.S[o] = mem.P[a].mean()
        elif name == "S_DOT": mem.S[o] = (mem.V[a] * mem.V[b]).sum()
        elif name == "V_ADD": mem.V[o] = mem.V[a] + mem.V[b]
        elif name == "V_SUB": mem.V[o] = mem.V[a] - mem.V[b]
        elif name == "V_MUL": mem.V[o] = mem.V[a] * mem.V[b]
        elif name == "V_SCALE": mem.V[o] = mem.V[a] * mem.S[b]
        elif name == "V_RELU": mem.V[o] = torch.relu(mem.V[a])
        elif name == "V_MATVEC_XT": mem.V[o] = (X.t() @ mem.P[a]) / N
        elif name == "P_MATVEC_X": mem.P[o] = X @ mem.V[a]
        elif name == "P_ADD": mem.P[o] = mem.P[a] + mem.P[b]
        elif name == "P_SUB": mem.P[o] = mem.P[a] - mem.P[b]
        elif name == "P_MUL": mem.P[o] = mem.P[a] * mem.P[b]
        elif name == "P_SCALE": mem.P[o] = mem.P[a] * mem.S[b]
        elif name == "P_ADDS": mem.P[o] = mem.P[a] + mem.S[b]
        elif name == "P_RELU": mem.P[o] = torch.relu(mem.P[a])
        elif name == "P_TANH": mem.P[o] = torch.tanh(mem.P[a])
        elif name == "P_LOADY": mem.P[o] = y
        # estabilidad numerica
        if t[1] == "S": mem.S[o] = _fix(mem.S[o])
        elif t[1] == "V": mem.V[o] = _fix(mem.V[o])
        elif t[1] == "P": mem.P[o] = _fix(mem.P[o])


# ----------------------------------------------------------------------------
# 2. EVALUACION DE UN ALGORITMO (entrena via Learn evolucionado -> R2 en val)
# ----------------------------------------------------------------------------
OUT_P = 0   # convencion: y_hat queda en el registro p0 tras Predict


def eval_algo(algo, tasks, T, nS, nV, nP, device):
    """Devuelve el R2 medio sobre las tareas dadas (entrenando desde cero cada una)."""
    r2s = []
    for (Xtr, ytr, Xval, yval) in tasks:
        try:
            F = Xtr.shape[1]
            mem = Mem(nS, nV, nP, F, Xtr.shape[0], device)
            run(algo["setup"], mem, Xtr, ytr, nS, nV, nP)
            for _ in range(T):
                run(algo["predict"], mem, Xtr, ytr, nS, nV, nP)   # escribe p0 = yhat_tr
                run(algo["learn"], mem, Xtr, ytr, nS, nV, nP)      # actualiza pesos
            # evaluacion en validacion: solo Predict, con los pesos aprendidos
            memv = Mem(nS, nV, nP, F, Xval.shape[0], device)
            memv.S = mem.S.clone(); memv.V = mem.V.clone()
            run(algo["predict"], memv, Xval, yval, nS, nV, nP)
            yhat = memv.P[OUT_P]
            if not torch.isfinite(yhat).all():
                r2s.append(-9.0); continue
            var = ((yval - yval.mean()) ** 2).mean() + 1e-12
            mse = ((yhat - yval) ** 2).mean()
            r2 = float(1.0 - mse / var)
            r2s.append(max(-9.0, min(1.0, r2)))
        except Exception:
            r2s.append(-9.0)
    return float(np.mean(r2s)), r2s


def learning_curve(algo, task, T, nS, nV, nP, device):
    """R2 en validacion vs numero de pasos de entrenamiento (muestra la adaptacion)."""
    Xtr, ytr, Xval, yval = task
    F = Xtr.shape[1]
    mem = Mem(nS, nV, nP, F, Xtr.shape[0], device)
    run(algo["setup"], mem, Xtr, ytr, nS, nV, nP)
    curve = []
    var = ((yval - yval.mean()) ** 2).mean() + 1e-12
    for step in range(T):
        run(algo["predict"], mem, Xtr, ytr, nS, nV, nP)
        run(algo["learn"], mem, Xtr, ytr, nS, nV, nP)
        memv = Mem(nS, nV, nP, F, Xval.shape[0], device)
        memv.S = mem.S.clone(); memv.V = mem.V.clone()
        run(algo["predict"], memv, Xval, yval, nS, nV, nP)
        yhat = memv.P[OUT_P]
        r2 = float(1.0 - ((yhat - yval) ** 2).mean() / var) if torch.isfinite(yhat).all() else -9.0
        curve.append(round(max(-9.0, min(1.0, r2)), 4))
    return curve


# ----------------------------------------------------------------------------
# 3. GENERACION / MUTACION / CROSSOVER DE ALGORITMOS
# ----------------------------------------------------------------------------
def rand_instr(opname, nS, nV, nP, rng):
    _, ot, i1, i2, uim = OP[opname]
    def ridx(tp):
        return int(rng.integers({"S": nS, "V": nV, "P": nP}[tp]))
    o = ridx(ot)
    a = ridx(i1) if i1 else 0
    b = ridx(i2) if i2 else 0
    imm = float(IMM_POOL[rng.integers(len(IMM_POOL))]) if uim else 0.0
    return (opname, o, a, b, imm)


def rand_fn(opset, length, nS, nV, nP, rng):
    return [rand_instr(opset[rng.integers(len(opset))], nS, nV, nP, rng) for _ in range(length)]


def enforce_output(predict, nS, nV, nP, rng):
    """La ultima instruccion de Predict debe escribir en p0 (convencion de salida)."""
    if not predict:
        predict = [rand_instr("P_MATVEC_X", nS, nV, nP, rng)]
    name, o, a, b, imm = predict[-1]
    if OP[name][1] != "P":                      # si no escribe P, forzamos una que si
        predict[-1] = rand_instr("P_MATVEC_X", nS, nV, nP, rng)
        name, o, a, b, imm = predict[-1]
    predict[-1] = (name, OUT_P, a, b, imm)
    return predict


def random_algo(nS, nV, nP, rng, ls=2, lp=3, ll=4):
    algo = {
        "setup": rand_fn(OPS_SETUP, int(rng.integers(0, ls + 1)), nS, nV, nP, rng),
        "predict": rand_fn(OPS_PREDICT, int(rng.integers(1, lp + 1)), nS, nV, nP, rng),
        "learn": rand_fn(OPS_LEARN, int(rng.integers(1, ll + 1)), nS, nV, nP, rng),
    }
    algo["predict"] = enforce_output(algo["predict"], nS, nV, nP, rng)
    return algo


def mutate(algo, nS, nV, nP, rng, maxlen=(4, 7, 9)):
    a = {k: list(v) for k, v in algo.items()}
    fn = rng.choice(["setup", "predict", "learn"])
    opset = {"setup": OPS_SETUP, "predict": OPS_PREDICT, "learn": OPS_LEARN}[fn]
    lim = {"setup": maxlen[0], "predict": maxlen[1], "learn": maxlen[2]}[fn]
    seq = a[fn]
    r = rng.random()
    if r < 0.4 and seq:                                  # modificar una instruccion
        i = int(rng.integers(len(seq)))
        seq[i] = rand_instr(seq[i][0] if rng.random() < 0.5
                            else opset[rng.integers(len(opset))], nS, nV, nP, rng)
    elif r < 0.7 and len(seq) < lim:                     # insertar
        seq.insert(int(rng.integers(len(seq) + 1)),
                   rand_instr(opset[rng.integers(len(opset))], nS, nV, nP, rng))
    elif r < 0.9 and len(seq) > (0 if fn == "setup" else 1):   # borrar
        del seq[int(rng.integers(len(seq)))]
    else:                                                # perturbar operandos/imm
        if seq:
            i = int(rng.integers(len(seq)))
            seq[i] = rand_instr(seq[i][0], nS, nV, nP, rng)
    a[fn] = seq
    a["predict"] = enforce_output(a["predict"], nS, nV, nP, rng)
    return a


def crossover(a1, a2, nS, nV, nP, rng):
    fn = rng.choice(["setup", "predict", "learn"])
    c = {k: list(v) for k, v in a1.items()}
    s2 = a2[fn]
    if s2:
        i = int(rng.integers(len(c[fn]) + 1))
        j = int(rng.integers(len(s2)))
        c[fn] = c[fn][:i] + [tuple(s2[j])] + c[fn][i:]
    c["predict"] = enforce_output(c["predict"], nS, nV, nP, rng)
    return c


# ----------------------------------------------------------------------------
# 4. ALGORITMO CODIFICADO A MANO (regresion lineal por gradiente) -> verificacion
# ----------------------------------------------------------------------------
def handcoded_linreg_gd(lr=0.5):
    # pesos en v0 ; lr en s0
    return {
        "setup": [("S_CONST", 0, 0, 0, lr)],                 # s0 = lr
        "predict": [("P_MATVEC_X", OUT_P, 0, 0, 0.0)],       # p0 = X @ v0
        "learn": [
            ("P_LOADY", 1, 0, 0, 0.0),                       # p1 = y
            ("P_SUB", 2, OUT_P, 1, 0.0),                     # p2 = yhat - y
            ("V_MATVEC_XT", 1, 2, 0, 0.0),                   # v1 = X^T p2 / N   (gradiente)
            ("V_SCALE", 1, 1, 0, 0.0),                       # v1 = lr * v1
            ("V_SUB", 0, 0, 1, 0.0),                         # v0 = v0 - v1
        ],
    }


# ----------------------------------------------------------------------------
# 5. TAREAS (distribucion de problemas de regresion)
# ----------------------------------------------------------------------------
def make_tasks(n_tasks, F, N, seed, kind="linear", noise=0.05):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        w = rng.normal(size=F)
        def gen(n):
            X = rng.normal(size=(n, F)).astype(np.float32)
            y = X @ w
            if kind == "relu":
                y = np.maximum(0.0, y) + 0.3 * (X[:, 0] * X[:, min(1, F - 1)])
            y = y + noise * rng.normal(size=n)
            y = (y - y.mean()).astype(np.float32)          # centrado (sin sesgo)
            return X, y
        Xtr, ytr = gen(N)
        Xval, yval = gen(N)
        tasks.append((Xtr, ytr, Xval, yval))
    return tasks


def to_device(tasks, device):
    out = []
    for Xtr, ytr, Xval, yval in tasks:
        out.append((torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device),
                    torch.tensor(Xval, device=device), torch.tensor(yval, device=device)))
    return out


# ----------------------------------------------------------------------------
# 6. IMPRESION LEGIBLE DE UN ALGORITMO
# ----------------------------------------------------------------------------
def fmt_instr(ins):
    name, o, a, b, imm = ins
    _, ot, i1, i2, uim = OP[name]
    reg = lambda tp, i: f"{tp.lower()}{i}"
    out = reg(ot, o)
    args = []
    if uim: args.append(f"{imm:g}")
    if i1: args.append(reg(i1, a))
    if i2: args.append(reg(i2, b))
    return f"    {out} = {name}({', '.join(args)})"


def print_algo(algo):
    for fn in ("setup", "predict", "learn"):
        print(f"  def {fn}:")
        if not algo[fn]:
            print("    (vacio)")
        for ins in algo[fn]:
            print(fmt_instr(ins))


# ----------------------------------------------------------------------------
# 7. BUCLE EVOLUTIVO (elitismo + torneo + mutacion/crossover + restart)
# ----------------------------------------------------------------------------
def evolve(train_tasks, test_tasks, device, nS=4, nV=4, nP=4, T=30,
           pop_size=150, generations=100, tournament=6, elite=8, cx_prob=0.4,
           restart_patience=15, time_budget=None, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    pop = [random_algo(nS, nV, nP, rng) for _ in range(pop_size)]
    hof = []                      # hall-of-fame: (fit, algo)
    best = None
    stall = 0
    prev_best = -1e9
    t0 = time.time()
    for gen in range(generations):
        fits = np.array([eval_algo(a, train_tasks, T, nS, nV, nP, device)[0] for a in pop])
        gi = int(np.argmax(fits))
        if best is None or fits[gi] > best[0]:
            best = (float(fits[gi]), {k: list(v) for k, v in pop[gi].items()})
        # hall-of-fame (mejores unicos por fitness)
        hof.append((float(fits[gi]), {k: list(v) for k, v in pop[gi].items()}))
        hof = sorted(hof, key=lambda z: -z[0])[:elite]

        if best[0] > prev_best + 1e-4:
            prev_best = best[0]; stall = 0
        else:
            stall += 1

        if verbose:
            te = eval_algo(best[1], test_tasks, T, nS, nV, nP, device)[0]
            print(f"gen {gen:3d} | best trainR2={best[0]:.4f} testR2={te:.4f} | genBest={fits[gi]:.4f} "
                  f"| stall={stall} | t={time.time()-t0:.1f}s", flush=True)

        if best[0] > 0.999 or (time_budget and time.time() - t0 > time_budget):
            break
        if gen == generations - 1:
            break

        order = np.argsort(-fits)
        new_pop = [pop[i] for i in order[:elite]]                  # ELITISMO
        if stall >= restart_patience:                              # RESTART (guardando HOF)
            new_pop = [dict(a) for _, a in hof]
            while len(new_pop) < pop_size:
                new_pop.append(random_algo(nS, nV, nP, rng))
            pop = new_pop[:pop_size]; stall = 0; prev_best = -1e9
            continue

        def tourney():
            idx = rng.integers(0, pop_size, size=tournament)
            return pop[idx[np.argmax(fits[idx])]]
        while len(new_pop) < pop_size:
            if rng.random() < cx_prob:
                child = crossover(tourney(), tourney(), nS, nV, nP, rng)
            else:
                child = mutate(tourney(), nS, nV, nP, rng)
            new_pop.append(child)
        pop = new_pop

    return best, hof


# ----------------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------------
def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, bf = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > bf:
            best, bf = i, free
    return torch.device(f"cuda:{best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="corre el algoritmo a mano y valida el VM")
    ap.add_argument("--kind", type=str, default="linear", choices=["linear", "relu"])
    ap.add_argument("--features", type=int, default=5)
    ap.add_argument("--npoints", type=int, default=256)
    ap.add_argument("--train-tasks", type=int, default=4)
    ap.add_argument("--test-tasks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30, help="pasos de entrenamiento por tarea (T)")
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    print(f"[dev] {device}")
    F, N, T = args.features, args.npoints, args.steps

    if args.verify:
        tasks = to_device(make_tasks(4, F, N, 123, args.kind), device)
        algo = handcoded_linreg_gd(lr=0.5)
        print("\n=== VM VERIFY: regresion lineal por gradiente (codificada a mano) ===")
        print_algo(algo)
        curve = learning_curve(algo, tasks[0], T, 4, 4, 4, device)
        r2, per = eval_algo(algo, tasks, T, 4, 4, 4, device)
        print(f"\ncurva de aprendizaje (R2 val por paso): {curve}")
        print(f"R2 medio en {len(tasks)} tareas: {r2:.4f}  (por tarea: {[round(x,3) for x in per]})")
        print("=> VM OK: el motor entrena de verdad." if r2 > 0.95 else "=> REVISAR: R2 bajo.")
        return

    print(f"[tareas] kind={args.kind} F={F} N={N} T={T} | train={args.train_tasks} test={args.test_tasks}")
    train_tasks = to_device(make_tasks(args.train_tasks, F, N, args.seed, args.kind), device)
    test_tasks = to_device(make_tasks(args.test_tasks, F, N, args.seed + 999, args.kind), device)

    print("\n=== EVOLUCION DE ALGORITMOS DE APRENDIZAJE (AutoML-Zero) ===")
    best, hof = evolve(train_tasks, test_tasks, device, T=T, pop_size=args.pop,
                       generations=args.generations, time_budget=args.time_budget, seed=args.seed)

    test_r2, test_per = eval_algo(best[1], test_tasks, T, 4, 4, 4, device)
    print(f"\n=== MEJOR ALGORITMO DESCUBIERTO ===")
    print(f"R2 train (tareas vistas)   : {best[0]:.4f}")
    print(f"R2 TEST  (tareas NUEVAS)   : {test_r2:.4f}   <- generalizacion (aprender a aprender)")
    print(f"R2 por tarea de test       : {[round(x,3) for x in test_per]}")
    print("\nPseudocodigo del algoritmo evolucionado:")
    print_algo(best[1])
    print("\ncurva de aprendizaje en una tarea de test (R2 val por paso):")
    print(" ", learning_curve(best[1], test_tasks[0], T, 4, 4, 4, device))


if __name__ == "__main__":
    main()
