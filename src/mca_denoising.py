"""
mca_denoising.py
==================================================================
Denoising 2D-1D para cubos espectrales mediante MORPHOLOGICAL
COMPONENT ANALYSIS (MCA / MMCA, estilo Starck-Elad-Donoho 2004/2005;
Bobin, Starck, Fadili & Moudden 2007).

La idea: un cubo s se descompone en una suma de COMPONENTES
MORFOLOGICAS, cada una sparse en un diccionario distinto. Aqui los
dos diccionarios COMPARTEN el eje espectral (misma wavelet 1D) y solo
difieren en el operador ESPACIAL:

    Phi_W = W_2D (x) W_1D    -> emision isotropa (puntual / extendida)
    Phi_C = C_2D (x) W_1D    -> estructuras anisotropas (curvilineas)

donde W_2D es una starlet 2D a trous (isotropa) y C_2D es la curvelet
UDCT (anisotropa). Compartir W_1D pone toda la diversidad morfologica
en el plano espacial, lo que hace separables las componentes.

La separacion se resuelve por BLOCK-COORDINATE RELAXATION (MMCA) con
un umbral decreciente segun la estrategia MOM (Mean-Of-Max): en cada
iteracion el umbral delta se fija a partir del residuo actual y baja
hasta un piso k_final*sigma. Esto da convergencia robusta sin afinar
a mano el calendario de umbrales (Bobin et al. 2007).

REUSO: toda la maquinaria 2D-1D (forward/inverse, wavelet 1D espectral,
soft-threshold, estimacion y calibracion de sigma por propagacion de
ruido) se importa de curvelets_denoising.py para garantizar que el
W_1D sea IDENTICO en ambos caminos. Solo se anade aqui la starlet 2D
isotropa con la MISMA interfaz que CurveletTransform2D.
==================================================================
"""

from __future__ import annotations
import numpy as np

from curvelets_denoising import (
    CurveletTransform2D,
    forward_2d1d,
    inverse_2d1d,
    soft_threshold,
    estimate_sigma_subband,
    estimate_global_sigma_from_negatives,
    calibrate_sigma_2d1d,
)


# ==================================================================
# 1. Starlet 2D isotropa (W_2D), interfaz identica a CurveletTransform2D
# ==================================================================
# A trous B3-spline NO decimada: cada subbanda conserva la forma del
# plano de entrada. forward devuelve una LISTA PLANA de subbandas 2D
# [detalle_1, ..., detalle_J, aproximacion]; inverse las suma para
# reconstruir el plano (reconstruccion a trous trivial). Esta interfaz
# permite reutilizar forward_2d1d / inverse_2d1d sin cambios.

def _atrous_conv_2d(plane: np.ndarray, step: int) -> np.ndarray:
    """
    Convolucion B3-spline a trous 2D (separable) con dilatacion `step`.
    Taps [1,4,6,4,1]/16 en offsets [-2,-1,0,1,2]*step, padding reflect.
    """
    h = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    offsets = np.array([-2, -1, 0, 1, 2]) * step
    pad = 2 * step

    # eje 0 (filas)
    n0 = plane.shape[0]
    xp = np.pad(plane, ((pad, pad), (0, 0)), mode="reflect")
    tmp = np.zeros_like(plane, dtype=float)
    for w, off in zip(h, offsets):
        tmp += w * xp[pad + off: pad + off + n0, :]

    # eje 1 (columnas)
    n1 = plane.shape[1]
    xp2 = np.pad(tmp, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(plane, dtype=float)
    for w, off in zip(h, offsets):
        out += w * xp2[:, pad + off: pad + off + n1]
    return out


class StarletTransform2D:
    """
    Starlet 2D a trous (B3-spline) NO decimada. forward devuelve una
    lista plana de subbandas 2D (misma shape que el plano); inverse
    reconstruye el plano sumando las subbandas.

    Parametros
    ----------
    shape : (ny, nx)
    nbscales : int | None
        Numero de planos de DETALLE. La lista resultante tiene
        nbscales+1 subbandas (detalles + aproximacion). Si None, se usa
        un valor por defecto en funcion del tamano del plano.
    """

    def __init__(self, shape, nbscales=None):
        self.shape = tuple(shape)
        ny, nx = self.shape
        if nbscales is None:
            nbscales = max(2, int(np.floor(np.log2(min(ny, nx)))) - 1)
        self.nbscales = int(nbscales)
        # subbanda de la aproximacion espacial (lowpass) en la lista plana:
        # forward devuelve [detalle_1, ..., detalle_J, aproximacion]
        self.lowpass_index = self.nbscales

    def forward(self, plane: np.ndarray) -> list[np.ndarray]:
        """plano 2D -> [detalle_1, ..., detalle_J, aproximacion]."""
        c = np.asarray(plane, dtype=float).copy()
        coeffs = []
        for j in range(self.nbscales):
            c_next = _atrous_conv_2d(c, 2 ** j)
            coeffs.append(c - c_next)   # plano de detalle
            c = c_next
        coeffs.append(c)                # aproximacion final
        return coeffs

    def inverse(self, flat: list[np.ndarray]) -> np.ndarray:
        """lista plana de subbandas -> plano 2D (suma a trous)."""
        out = None
        for sub in flat:
            out = np.asarray(sub) if out is None else out + np.asarray(sub)
        return np.asarray(out).real


# ==================================================================
# 2. Calibracion de sigma POR DICCIONARIO
# ==================================================================
# Cada diccionario (starlet y curvelet) colorea el ruido de forma
# distinta (la curvelet UDCT es un tight frame muy redundante; la
# starlet 2D es no-decimada), asi que cada uno tiene su propio mapa de
# sigma por (subbanda, escala espectral). Se propaga la MISMA
# realizacion de ruido por cada diccionario via calibrate_sigma_2d1d.

def calibrate_sigma_per_dict(shape, transforms, n_spec_scales, sigma_global,
                             n_real: int = 1, seed: int = 0):
    """
    Devuelve una lista de sigma_map (uno por transformada en `transforms`),
    cada uno con el formato de calibrate_sigma_2d1d: lista por subbanda de
    np.ndarray (n_spec_scales+1,) con sigma por escala espectral.
    """
    return [
        calibrate_sigma_2d1d(shape, t, n_spec_scales, sigma_global,
                             n_real=n_real, seed=seed)
        for t in transforms
    ]


def _sigma_map_from_data(cube, transform, n_spec_scales, s_factor: float = 1.0):
    """sigma por subbanda via MAD sobre los coeficientes de los datos."""
    coeffs = forward_2d1d(cube, transform, n_spec_scales)
    return [np.array([estimate_sigma_subband(w1d[s], s_factor=s_factor)
                      for s in range(w1d.shape[0])])
            for w1d in coeffs]


# ==================================================================
# 3. Umbral MOM (Mean-Of-Max) y umbralizado por subbanda
# ==================================================================

def _mom_threshold(residual, transforms, sigma_maps, n_spec_scales) -> float:
    """
    Umbral MOM (crudo, SIN piso) a partir del residuo total E
    (Bobin et al. 2007):

        delta = mean_k ( || (Phi_k^T E) / sigma_{k,b} ||_inf )

    El maximo se toma en UNIDADES DE SIGMA (coeficiente / sigma de su
    subbanda), de modo que delta es directamente el nivel de
    significancia del coeficiente mas fuerte que aun queda por extraer.
    El piso (k_final) se aplica fuera, ahora POR DICCIONARIO.
    """
    maxes = []
    for t, smap in zip(transforms, sigma_maps):
        lowpass = getattr(t, "lowpass_index", None)
        coeffs = forward_2d1d(residual, t, n_spec_scales)
        m = 0.0
        for b, w1d in enumerate(coeffs):
            if b == lowpass:
                continue                      # excluye lowpass espacial (continuo)
            n_spec = w1d.shape[0]
            for s in range(n_spec):
                if s == n_spec - 1:
                    continue                  # excluye aproximacion espectral
                sig = smap[b][s]
                if not np.isfinite(sig) or sig <= 0:
                    continue
                v = float(np.abs(w1d[s]).max()) / sig
                if v > m:
                    m = v
        maxes.append(m)
    return float(np.mean(maxes)) if maxes else 0.0


def _threshold_subbands(coeffs, delta, sigma_map, coeffs_model=None,
                        eps: float = 1e-33):
    """
    Soft-threshold por subbanda con umbral delta*sigma (delta en unidades
    de sigma). Si se pasa `coeffs_model` (descomposicion del modelo previo
    de ESTA componente en ESTE diccionario) se aplica reweighted L1
    (Candes, Wakin & Boyd 2008): los coeficientes grandes del modelo
    reciben peso ~0 (umbral ~0, menos sesgo de encogimiento) y los
    pequenos / ausentes reciben peso 1 (umbral = base, NUNCA mayor).

    NOTA: el peso se CAPA en 1.0 (thr <= base). Sin este tope, un modelo
    casi vacio (|model|~0) genera weights = base/eps astronomicos que
    matan cualquier coeficiente nuevo; combinado con el calendario
    descendente (el modelo entra vacio al primer reweight) el modelo se
    queda clavado en su estado inicial y el denoising no avanza. El
    reweighted-L1 solo debe REDUCIR el umbral en coeficientes fuertes ya
    presentes en el modelo, nunca aumentarlo por encima del piso base.
    """
    out = []
    for b, w1d in enumerate(coeffs):
        w_new = w1d.copy()
        for s in range(w1d.shape[0]):
            sig = sigma_map[b][s]
            base = delta * sig
            if coeffs_model is None:
                thr = base
            else:
                model_sub = coeffs_model[b][s]
                # peso de Candes capado en 1: solo des-sesga coeficientes
                # grandes del modelo (peso < 1), nunca endurece el umbral.
                weights = np.minimum(base / (np.abs(model_sub) + eps), 1.0)
                mask = np.abs(w1d[s]) > base
                thr = np.where(mask, weights * base, base)
            w_new[s] = soft_threshold(w1d[s], thr)
        out.append(w_new)
    return out


def _extract_coarse_2d1d(cube, transform, n_spec_scales, n_chan):
    """
    Aproximacion 2D-1D mas gruesa: termino (subbanda espacial coarse) x
    (aproximacion espectral). Para la starlet 2D la subbanda coarse es la
    ultima de la lista plana, y la aproximacion espectral es el ultimo
    plano del eje 0. Este termino NO se umbraliza (continuo / flujo).
    """
    coeffs = forward_2d1d(cube, transform, n_spec_scales)
    coarse = [np.zeros_like(w) for w in coeffs]
    coarse[-1][-1] = coeffs[-1][-1]      # espacial coarse x espectral approx
    return inverse_2d1d(coarse, transform, n_chan)


def _curvelet_drop_indices(tC, min_scale: int) -> list[int]:
    """
    Indices (en la lista plana de subbandas) de las escalas curvelet MAS
    GRUESAS que `min_scale`, que se EXCLUYEN de la componente anisotropa.

    La UDCT ordena la lista plana por escala creciente: escala 0 es el
    lowpass isotropo, escala 1 la direccional mas gruesa, etc. Sus wedges
    gruesos son atomos no-locales y oscilantes: si la curvelet los usa para
    representar estructura de gran escala (que corresponde a la starlet),
    reconstruyen un patron global ondulado (anillamiento). Confinando la
    curvelet a escalas >= min_scale, las bajas frecuencias quedan para la
    starlet + s_coarse y la curvelet solo codifica bordes/filamentos.
    """
    struct = getattr(tC, "_struct", None)
    if struct is None:
        return []
    drop = []
    flat = 0
    for si, scale_meta in enumerate(struct):
        n = int(sum(scale_meta))
        if si < min_scale:
            drop.extend(range(flat, flat + n))
        flat += n
    return drop


# ==================================================================
# 4. Bucle MCA / MMCA (Block-Coordinate Relaxation + MOM)
# ==================================================================

def mca_denoise_cube(
    cube: np.ndarray,
    n_spec_scales: int = 3,
    n_spatial_scales: int | None = None,
    k_final: float = 3.0,
    k_final_W: float | None = None,
    k_final_C: float | None = None,
    max_iter: int = 30,
    schedule: str = "linear",
    floor_iters: int = 5,
    curvelet_min_scale: int = 1,
    positivity: str = "estimate",
    plateau_tol: float = 1e-4,
    plateau_patience: int = 3,
    ct_kwargs: dict | None = None,
    reweight: bool = False,
    eps: float = 1e-33,
    reference: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
    ref_support_frac: float | None = None,
    noise_cube: np.ndarray | None = None,
    sigma_global: float | None = None,
    calibrate: bool = True,
    calib_realizations: int = 1,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Denoising por Morphological Component Analysis (MCA / MMCA).

    Block-Coordinate Relaxation con umbral decreciente: por defecto el
    umbral delta baja linealmente desde lambda_max (fijado por MOM sobre
    el residuo inicial) hasta el piso k_final (MCA clasico de
    Starck-Elad-Donoho); con schedule='mom' se recalcula delta de forma
    adaptativa cada iteracion (Bobin et al. 2007).

    Separa el cubo en una componente ISOTROPA (s_W, diccionario starlet
    2D x wavelet 1D) y otra ANISOTROPA (s_C, diccionario curvelet 2D x
    wavelet 1D), mas un continuo s_coarse fijo. El denoising es
    estimate = s_coarse + s_W + s_C.

    Parametros
    ----------
    cube : (n_chan, ny, nx)
        Cubo ruidoso (eje 0 espectral).
    n_spec_scales : int
        Escalas de la wavelet 1D espectral (compartida por ambos dicts).
    n_spatial_scales : int | None
        Planos de detalle de la starlet 2D (W_2D). None -> automatico.
    k_final : float
        Piso del umbral en unidades de sigma (tipico 3-4). Valor por
        defecto para ambos diccionarios.
    k_final_W, k_final_C : float | None
        Piso del umbral POR DICCIONARIO (starlet isotropa / curvelet). Si
        None usan `k_final`. La curvelet UDCT es un frame muy redundante,
        asi que suele necesitar un piso mas alto (p.ej. k_final_C=4-4.5)
        para no dejar pasar grano de ruido, mientras la starlet representa
        de forma compacta la emision isotropa y tolera un piso menor
        (k_final_W=3).
    max_iter : int
        Tope de iteraciones externas.
    schedule : {'linear', 'mom'}
        Calendario del umbral delta (en unidades de sigma):
          - 'linear' (recomendado, MCA clasico de Starck-Elad-Donoho):
            delta desciende linealmente desde lambda_max (fijado por MOM
            sobre el residuo inicial) hasta el piso k_final. Garantiza
            alcanzar el suelo de ruido y denoisar de verdad.
          - 'mom' (Bobin et al. 2007): delta se recalcula en cada
            iteracion como la media de los maximos normalizados del
            residuo (max(delta, k_final)). Es adaptativo pero puede
            estancarse por encima del piso en cubos con pocas fuentes.
    floor_iters : int
        Iteraciones finales (schedule='linear') que se mantienen EN el piso
        k_final para refinar: cada paso a umbral bajo fijo es un IST que
        recupera progresivamente la estructura fina, en vez de tocar el
        piso una sola vez al final.
    curvelet_min_scale : int
        Escala curvelet mas gruesa que SI entra en s_C. 0 = usa todas
        (incluido el lowpass isotropo y los wedges gruesos, que tienden a
        producir anillamiento global). 1 (recomendado) = excluye el lowpass
        y deja las bajas frecuencias a la starlet + s_coarse. 2 = excluye
        ademas la escala direccional mas gruesa (mas agresivo contra el
        anillamiento, a costa de bordes de gran escala).
    positivity : {'estimate', 'component', 'none'}
        Como se impone flujo >= 0:
          - 'estimate' (recomendado): NO se rectifica cada componente
            durante la iteracion; la no-negatividad se aplica al estimate
            total al final. Asi s_C conserva lobulos +/- (corrige
            oscilaciones) en vez de rectificarse en una rejilla de bultos
            con sesgo positivo.
          - 'component': rectifica cada componente en cada iteracion
            (comportamiento clasico; puede sesgar s_C).
          - 'none': sin positividad.
    plateau_tol, plateau_patience :
        Criterio de parada por meseta de Var(residuo) (solo se aplica
        una vez delta ha alcanzado el piso k_final).
    ct_kwargs : dict | None
        kwargs para CurveletTransform2D (p.ej. nbscales, nbangles_coarse).
    reweight : bool
        Activa reweighted L1 (pesos del modelo previo de cada componente).
    reference : (n_chan, ny, nx) | None
        Cubo limpio para seleccionar el mejor modelo por RMSE. Sin
        referencia se devuelve el modelo convergido (ultimo).
    ref_mask : np.ndarray | None
        Mascara booleana (broadcastable al cubo) que restringe el RMSE de
        seleccion al SOPORTE de la senal. Con senal dispersa, el RMSE
        GLOBAL premia el estimate vacio (el fondo de ceros domina la media
        y cualquier extraccion real anade ruido que sube el RMSE), de modo
        que el "mejor" modelo acaba siendo ~0. Midiendo el RMSE solo donde
        hay senal, el modelo vacio tiene error alto (no recupera el arco) y
        recuperar la estructura baja el error. Si None y se da
        `ref_support_frac`, la mascara se deriva de `reference`.
    ref_support_frac : float | None
        Fraccion del maximo de |reference| para derivar `ref_mask`
        automaticamente: soporte = |reference| > frac * |reference|.max().
        Tipico 0.01-0.1. Ignorado si se pasa `ref_mask` explicita.
    noise_cube : np.ndarray | None
        Realizacion de ruido a propagar para calibrar sigma (manda sobre
        sigma_global si se da).
    sigma_global : float | None
        Sigma global a propagar (si None y calibrate, se estima de la cola
        negativa del cubo).
    calibrate : bool
        Calibra sigma por propagacion de ruido (recomendado). Si False, se
        estima sigma por subbanda via MAD sobre los datos.
    calib_realizations : int
        Realizaciones Monte-Carlo para promediar sigma.

    Devuelve
    --------
    dict con claves:
        'estimate' : cubo denoised (s_coarse + s_W + s_C)
        's_W'      : componente isotropa
        's_C'      : componente anisotropa
        's_coarse' : continuo (no umbralizado)
        'residual' : cube - estimate
    """
    ct_kwargs = ct_kwargs or {}
    cube = np.asarray(cube, dtype=float)
    n_chan, ny, nx = cube.shape

    # --- diccionarios espaciales (comparten W_1D via forward_2d1d) ---
    tW = StarletTransform2D(shape=(ny, nx), nbscales=n_spatial_scales)
    tC = CurveletTransform2D(shape=(ny, nx), **ct_kwargs)
    tC.lowpass_index = 0   # en la UDCT la subbanda lowpass es la primera (b==0)
    transforms = [tW, tC]
    names = ["W (isotropa)", "C (anisotropa)"]

    # (fix 1) confinar la curvelet a escalas finas: subbandas a anular en s_C
    tW.drop_indices = []
    tC.drop_indices = _curvelet_drop_indices(tC, curvelet_min_scale)
    if verbose and tC.drop_indices:
        print(f"(*) curvelet confinada a escalas >= {curvelet_min_scale}: "
              f"se excluyen {len(tC.drop_indices)} subbandas gruesas de s_C.")

    # piso del umbral POR DICCIONARIO (en unidades de sigma)
    kf = [float(k_final_W if k_final_W is not None else k_final),
          float(k_final_C if k_final_C is not None else k_final)]

    # --- calibracion de sigma por diccionario ---
    if calibrate:
        if noise_cube is not None:
            sigma_maps = [
                [np.array([estimate_sigma_subband(w1d[s], s_factor=1.0)
                           for s in range(w1d.shape[0])])
                 for w1d in forward_2d1d(noise_cube, t, n_spec_scales)]
                for t in transforms
            ]
        else:
            if sigma_global is None:
                sigma_global = estimate_global_sigma_from_negatives(cube)
            sigma_maps = calibrate_sigma_per_dict(
                (n_chan, ny, nx), transforms, n_spec_scales, sigma_global,
                n_real=calib_realizations, seed=seed,
            )
        if verbose:
            sg = sigma_global if sigma_global is not None else float("nan")
            print(f"(*) Calibracion de ruido: sigma_global={sg:.4e} | "
                  f"{len(sigma_maps)} diccionarios calibrados "
                  f"({[len(sm) for sm in sigma_maps]} subbandas).")
    else:
        sigma_maps = [_sigma_map_from_data(cube, t, n_spec_scales)
                      for t in transforms]

    # --- continuo fijo (aproximacion 2D-1D mas gruesa, no umbralizado) ---
    s_coarse = _extract_coarse_2d1d(cube, tW, n_spec_scales, n_chan)
    s_coarse = np.maximum(0.0, s_coarse)

    # --- inicializacion de las componentes ---
    components = [np.zeros_like(cube) for _ in transforms]

    use_reference = reference is not None
    prev_resid_var = np.inf
    stall = 0

    best_components = [c.copy() for c in components]
    best_score = np.inf
    best_iter = -1

    # --- mascara de soporte para el RMSE de seleccion ---
    # Sin mascara, el RMSE global premia el estimate vacio cuando la senal
    # es dispersa. Restringiendo el RMSE al soporte, el modelo vacio tiene
    # error alto y recuperar la estructura baja el error.
    sel_mask = None
    if use_reference:
        if ref_mask is not None:
            sel_mask = np.broadcast_to(np.asarray(ref_mask, dtype=bool),
                                       reference.shape)
        elif ref_support_frac is not None:
            ref_abs = np.abs(reference)
            ref_peak = float(ref_abs.max())
            if ref_peak > 0:
                sel_mask = ref_abs > ref_support_frac * ref_peak
        if sel_mask is not None and not sel_mask.any():
            sel_mask = None  # mascara vacia -> RMSE global (fallback)
        if verbose:
            if sel_mask is not None:
                print(f"(*) Seleccion por RMSE en soporte: "
                      f"{int(sel_mask.sum())} voxeles "
                      f"({100.0 * sel_mask.mean():.2f}% del cubo).")
            else:
                print("(*) Seleccion por RMSE GLOBAL (sin mascara de "
                      "soporte; con senal dispersa puede premiar el "
                      "estimate vacio).")

    def _rmse(a, b):
        if sel_mask is not None:
            return float(np.sqrt(np.mean((a[sel_mask] - b[sel_mask]) ** 2)))
        return float(np.sqrt(np.mean((a - b) ** 2)))

    # lambda_max para el calendario: MOM sobre el residuo inicial. Es el
    # umbral comun de arranque; debe quedar por encima de todos los pisos.
    E0 = cube - s_coarse
    lambda_max = _mom_threshold(E0, transforms, sigma_maps, n_spec_scales)
    lambda_max = max(lambda_max, max(kf))
    if verbose and schedule == "linear":
        print(f"(*) lambda_max (MOM inicial)={lambda_max:.3f} -> "
              f"desciende a k_final por dict {kf} en {max_iter} iter.")

    for it in range(max_iter):
        # (1) umbral delta POR DICCIONARIO segun el calendario
        E = cube - s_coarse - sum(components)
        if schedule == "mom":
            mom_raw = _mom_threshold(E, transforms, sigma_maps, n_spec_scales)
            deltas = [max(mom_raw, kf[k]) for k in range(len(transforms))]
            at_floor = all(deltas[k] <= kf[k] + 1e-9 for k in range(len(transforms)))
        else:  # 'linear': desciende lambda_max -> k_final_k y refina en el piso
            n_desc = max(1, max_iter - max(0, floor_iters))
            frac = min(1.0, it / max(1, n_desc - 1))
            deltas = [max((1.0 - frac) * lambda_max + frac * kf[k], kf[k])
                      for k in range(len(transforms))]
            at_floor = (it >= n_desc - 1)

        # (2) actualizacion por bloques (Gauss-Seidel)
        for k, (t, smap) in enumerate(zip(transforms, sigma_maps)):
            # residuo marginal: usa las componentes ya actualizadas (j != k)
            r_k = cube - s_coarse - sum(components[j] for j in range(len(components)) if j != k)
            coeffs = forward_2d1d(r_k, t, n_spec_scales)
            coeffs_model = None
            if reweight and it > 0:
                coeffs_model = forward_2d1d(components[k], t, n_spec_scales)
            coeffs_t = _threshold_subbands(coeffs, deltas[k], smap,
                                           coeffs_model=coeffs_model, eps=eps)
            for idx in t.drop_indices:        # (fix 1) anula escalas gruesas
                coeffs_t[idx][:] = 0.0
            s_k = inverse_2d1d(coeffs_t, t, n_chan)
            if positivity == "component":     # (fix 4) positividad por comp.
                s_k = np.maximum(0.0, s_k)
            components[k] = s_k

        estimate = s_coarse + sum(components)
        if positivity == "estimate":
            estimate = np.maximum(0.0, estimate)
        resid_var = float(np.var(cube - estimate))
        rel_change = abs(prev_resid_var - resid_var) / (prev_resid_var + 1e-30)

        # (3) seleccion del mejor modelo
        if use_reference:
            score = _rmse(estimate, reference)
            if score < best_score:
                best_score = score
                best_components = [c.copy() for c in components]
                best_iter = it
        else:
            best_components = [c.copy() for c in components]
            best_iter = it
            best_score = resid_var

        if verbose:
            extra = f" | rmse_ref={best_score:.6e}" if use_reference else ""
            print(f"iter {it:2d} | deltaW={deltas[0]:6.3f} deltaC={deltas[1]:6.3f} "
                  f"| resid_var={resid_var:.6e} | rel_change={rel_change:.2e}{extra}")

        # parada por meseta SOLO cuando delta ya alcanzo el piso k_final
        # (asi el calendario decreciente no se corta antes de llegar al
        # suelo de ruido)
        if at_floor and rel_change < plateau_tol:
            stall += 1
            if stall >= plateau_patience:
                if verbose:
                    print(f"Plateau alcanzado en iter {it}.")
                break
        else:
            stall = 0
        prev_resid_var = resid_var

    if verbose:
        crit = "rmse_ref" if use_reference else "resid_var"
        print(f"Mejor modelo: iter {best_iter} ({crit}={best_score:.6e}).")
        for nm, c in zip(names, best_components):
            print(f"    componente {nm}: flujo={float(c.sum()):.4e}")

    estimate = s_coarse + sum(best_components)
    if positivity == "estimate":
        estimate = np.maximum(0.0, estimate)
    return {
        "estimate": estimate,
        "s_W": best_components[0],
        "s_C": best_components[1],
        "s_coarse": s_coarse,
        "residual": cube - estimate,
    }


if __name__ == "__main__":
    print(__doc__)
    print("Uso: mca_denoise_cube(cube) sobre un cubo (n_chan, ny, nx). "
          "Devuelve dict con 'estimate', 's_W', 's_C', 's_coarse', 'residual'.")
