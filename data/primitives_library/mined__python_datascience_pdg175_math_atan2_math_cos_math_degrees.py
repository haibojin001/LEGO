# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg175::math.atan2+math.cos+math.degrees
# name: math_primitive
# summary: Uses math.atan2, math.cos, math.degrees, math.pow across 2 repos
# anchor_symbols: ['math.atan2', 'math.cos', 'math.degrees', 'math.pow', 'math.radians', 'math.sin']
# observed in 2 repos: ['jasmcaus__caer', 'mwaskom__seaborn']...

# --- from mwaskom__seaborn::seaborn/external/husl.py::lch_to_luv ---
def lch_to_luv(triple):
    L, C, H = triple

    Hrad = math.radians(H)
    U = (math.cos(Hrad) * C)
    V = (math.sin(Hrad) * C)

    return [L, U, V]

# --- from mwaskom__seaborn::seaborn/external/husl.py::luv_to_lch ---
def luv_to_lch(triple):
    L, U, V = triple

    C = (math.pow(math.pow(U, 2) + math.pow(V, 2), (1.0 / 2.0)))
    hrad = (math.atan2(V, U))
    H = math.degrees(hrad)
    if H < 0.0:
        H = 360.0 + H

    return [L, C, H]

# --- from jasmcaus__caer::caer/transforms/color.py::_get_affine_matrix ---
def _get_affine_matrix(center, angle, translate, scale, shear) -> Tensor:
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]

    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale

    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    RSS = np.array(
        [[math.cos(angle) * scale, -math.sin(angle + shear) * scale, 0],
         [math.sin(angle) * scale,
          math.cos(angle + shear) * scale, 0], [0, 0, 1]])
    matrix = T @ C @ RSS @ np.linalg.inv(C)

    return matrix[:2, :]
