#!/public/home/xinghy/anaconda3-2023.03/bin/python
import os
import sys
import numpy as np
import time
from lattice.insertion.gamma import gamma
from opt_einsum import contract

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, "/public/home/gengyq/EasyDistillation"))

from lattice import set_backend, get_backend

set_backend("cupy")

Lt = 64
Lx = 32
conf_id = 58150
###############################################################################
from lattice.insertion.mom_dict import momDict_mom1
from lattice.insertion import (
    Insertion,
    Operator,
    GammaName,
    DerivativeName,
    ProjectionName,
)

ins_P = Insertion(GammaName.B1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom1)
op_P = Operator("proton", [ins_P[1](0, 0, 0)], [1])
print("ins_P", ins_P.rows)
###############################################################################
from lattice import preset

elemental = preset.ElementalNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".VVV.npy",
    [1, 6, 64, 100, 100, 100],  # Nder,Nmom,Nt,Ne,Ne,Ne
    100,
)
perambulator_u = preset.PerambulatorNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".peramb.light.npy",
    [64, 64, 4, 4, 100, 100],
    100,
)
perambulator_s = preset.PerambulatorNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".peramb.strange.npy",
    [64, 64, 4, 4, 100, 100],
    100,
)
###############################################################################
from lattice.quark_diagram import (
    BaryonDiagram,
    compute_diagrams_multitime,
    Baryon,
    Propagator,
)


peramb_u = Propagator(perambulator_u, Lt)
peramb_s = Propagator(perambulator_s, Lt)
###############################################################################

peramb_u.load(conf_id)
peramb_s.load(conf_id, 2)

P_P1 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    None,
)

P_P2 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 4, 2, 2, 3, 3],
)

P_P3 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 1, 2, 2, 3, 6],
)

P_P4 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 4, 2, 2, 3, 6],
)

# P_P5 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 1, 2, 5, 3, 3],
# )

P_P6 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 4, 2, 5, 3, 3],
)

# P_P7 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 1, 2, 5, 3, 6],
# )

P_P8 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 4, 2, 5, 3, 6],
)

print("diagram set done")

###############################################################################
# t_snk = np.arange(Lt)

backend = get_backend()
twopt = backend.zeros((6, Lt, 2, 2, 2, 2), "<c16")

for p1 in range(6):
    st2 = time.time()
    for p2 in range(6):
        P_src_1 = Baryon(elemental[p1], op_P, True)
        P_src_2 = Baryon(elemental[int((p1 + 3) % 6)], op_P, True)
        P_snk_1 = Baryon(elemental[p2], op_P, False)
        P_snk_2 = Baryon(elemental[int((p2 + 3) % 6)], op_P, False)
        P_src_1.load(conf_id)
        P_snk_1.load(conf_id)
        P_src_2.load(conf_id)
        P_snk_2.load(conf_id)

        for t_src in range(1):
            st1 = time.time()
            # peram_all_light[t_src] = np.roll(peram_all_light[t_src], t_src, 0)
            for t_snk in range(Lt):
                tmp = compute_diagrams_multitime(
                    [P_P1, P_P2, P_P3, P_P4, P_P6, P_P8],
                    [t_snk, t_src, t_snk, t_src],
                    [P_snk_1, P_src_1, P_snk_2, P_src_2],
                    [None, peramb_u, peramb_u, peramb_s],
                    "",
                )
                # if t_snk < t_src:
                # tmp = -tmp
                twopt[:, (t_snk - t_src) % Lt] += tmp

                # del peram_u
                # cp._default_memory_pool.free_allsa_blocks()

            ed1 = time.time()
            print(
                f"p1{p1}p2{p2} time{t_src} caululate done, time used: %.3f s"
                % (ed1 - st1)
            )
    ed2 = time.time()
    print(f"all p1{p1}p2{p2} caululate done, time used: %.3f s" % (ed2 - st2))

pp = backend.asarray((gamma(0) + gamma(8)) / 2)[0:2, 0:2]
pm = backend.asarray((gamma(0) - gamma(8)) / 2)[0:2, 0:2]
corr_diagram = contract("ntqryz,Qq,Rr,Yy,Zz->ntQRYZ", twopt, pp, pp, pp, pp)

np.save(
    "/public/home/gengyq/laph/Lambda/result/di_lambda_diagram_dirac_t0_p1.npy",
    corr_diagram,
)

corr_none_dirac = np.zeros((6, Lt), "<c16")
for id1 in range(2):
    if id1 == 0:
        id2 = 1
        sign1 = 1.0
    else:
        id2 = 0
        sign1 = -1.0
    for id3 in range(2):
        if id3 == 0:
            id4 = 1
            sign2 = 1.0
        else:
            id4 = 0
            sign2 = -1.0
            corr_none_dirac += sign1 * sign2 * corr_diagram[:, :, id1, id3, id2, id4]
# print(corr_diagram[:, 0])

di_baryon = np.zeros(Lt, "<c16")
di_baryon = (
    -corr_none_dirac[0]
    + 2 * corr_none_dirac[1]
    + corr_none_dirac[2]
    - 2 * corr_none_dirac[3]
    - corr_none_dirac[5]
    + corr_none_dirac[7]
)
print(di_baryon)
np.save("/public/home/gengyq/laph/Lambda/result/di_lambda_corr_t0_p1.npy", di_baryon)
