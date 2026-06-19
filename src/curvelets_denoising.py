"""
curvelets_denoising.py
==================================================================
Denoising 2D-1D para cubos espectrales, sustituyendo el operador
ESPACIAL wavelet por una transformada CURVELET 2D, mientras el eje
ESPECTRAL se mantiene en wavelet 1D.

    W_2D (x) W_1D    -->    C_2D (x) W_1D

Motivacion: las wavelets 2D separables son sub-optimas para
singularidades a lo largo de curvas (brazos, bordes, filamentos);
las curvelets alcanzan aproximacion cuasi-optima para objetos C^2
con bordes C^2 (Candes & Donoho 2004, CPAM 57, 219). El esquema
2D-1D respeta que el plano espacial y el eje espectral tienen
estructura fisica distinta (Starck, Fadili & Rassat 2009, A&A 504,
641). El eje espectral NO se cambia: el perfil de linea es una
singularidad 1D bien representada por wavelets.

Envoltura IST con soft-thresholding (estilo Lahiry et al. 2026,
arXiv:2602.10893; reweighted L1 -> Candes, Wakin & Boyd 2008).

DEPENDENCIA CLAVE para la curvelet:
  - Opcion A (recomendada): PyCurvelab / curvelops (wrapper Python
    sobre CurveLab de Candes, Demanet, Donoho, Ying).
  - Opcion B: la curvelet de PySAP / Sparse2D (la que ya usas).
  El codigo abajo abstrae la transformada tras una interfaz
  CurveletTransform2D para que enganches la implementacion que
  tengas instalada sin cambiar la logica 2D-1D.
==================================================================
"""

from __future__ import annotations
import numpy as np

from curvelets.numpy import UDCT


# ==================================================================
# 1. Interfaz a la curvelet 2D (backend: curvelets / UDCT)
# ==================================================================
# Se abstrae para que la maquinaria 2D-1D no dependa del backend.
# La transformada devuelve una LISTA PLANA de subbandas; cada subbanda
# es un array 2D de coeficientes (escala + orientacion). inverse las
# recombina al plano.
#
# Backend: paquete `curvelets` (Uniform Discrete Curvelet Transform,
# Python puro, MIT). Su estructura nativa es anidada
#   coeffs[escala][direccion][angulo] -> array 2D
# que aqui se APLANA a una lista de subbandas y se RECONSTRUYE usando
# una plantilla de estructura calculada una sola vez.

class CurveletTransform2D:
    """
    Wrapper sobre la UDCT del paquete `curvelets`. forward devuelve una
    lista plana de subbandas 2D; inverse reconstruye el plano.

    Parametros
    ----------
    shape : (ny, nx)
    nbscales : int | None
        Numero total de escalas (incluye la lowpass). UDCT requiere >=2.
        None -> usa el valor por defecto del backend (3).
    nbangles_coarse : int
        Cuñas angulares por direccion en la escala mas gruesa (UDCT
        requiere >=3; el numero se duplica al refinar). Antes ligado al
        sentido FDCT; aqui se mapea a `wedges_per_direction`.
    transform_kind : {'real', 'complex'}
        'real' es lo adecuado para cubos de datos reales.
    """

    def __init__(self, shape, nbscales=None, nbangles_coarse=3,
                 transform_kind="real"):
        self.shape = tuple(shape)
        self.nbscales = nbscales
        self.nbangles_coarse = nbangles_coarse

        kwargs = {"wedges_per_direction": max(3, int(nbangles_coarse))}
        if nbscales is not None:
            kwargs["num_scales"] = int(nbscales)
        self._op = UDCT(self.shape, transform_kind=transform_kind, **kwargs)

        # Plantilla de estructura (nº de angulos por escala/direccion)
        # para reconstruir la jerarquia desde la lista plana.
        template = self._op.forward(np.zeros(self.shape, dtype=float))
        self._struct = [[len(direction) for direction in scale]
                        for scale in template]

    def forward(self, plane: np.ndarray) -> list[np.ndarray]:
        """plano 2D -> lista plana de subbandas (cada una array 2D)."""
        coeffs = self._op.forward(np.asarray(plane, dtype=float))
        flat = []
        for scale in coeffs:
            for direction in scale:
                for ang in direction:
                    flat.append(np.asarray(ang))
        return flat

    def inverse(self, flat: list[np.ndarray]) -> np.ndarray:
        """lista plana de subbandas -> plano 2D reconstruido (real)."""
        it = iter(flat)
        coeffs = []
        for scale_meta in self._struct:
            scale = []
            for n_ang in scale_meta:
                scale.append([next(it) for _ in range(n_ang)])
            coeffs.append(scale)
        rec = self._op.backward(coeffs)
        return np.asarray(rec).real


# ==================================================================
# 2. Wavelet 1D a lo largo del eje espectral
# ==================================================================
# Starlet/IUWT 1D (a trous) -- consistente con la familia de
# transformadas no-decimadas que usas. Se aplica POR cada coeficiente
# curvelet, a lo largo del eje espectral.

def starlet_1d_forward(signal: np.ndarray, n_scales: int) -> np.ndarray:
    """
    Transformada a trous (B3-spline) a lo largo del EJE 0 (espectral).

    Es no-decimada: cada plano conserva la longitud del eje espectral.
    Acepta arrays N-dimensionales (el eje 0 es el espectral; los demas
    son espaciales) y devuelve un array de shape
    (n_scales+1, *signal.shape): n_scales planos de detalle + 1 de
    aproximacion.
    """
    # preserva dtype complejo (los coeficientes curvelet son complejos)
    c = np.asarray(signal, dtype=np.result_type(signal, float)).copy()
    coeffs = []
    for j in range(n_scales):
        c_next = _atrous_conv_axis0(c, 2 ** j)
        coeffs.append(c - c_next)   # plano de detalle
        c = c_next
    coeffs.append(c)                # aproximacion final
    return np.stack(coeffs, axis=0)


def starlet_1d_inverse(coeffs: np.ndarray) -> np.ndarray:
    """Reconstruccion a trous: suma de todos los planos (eje 0)."""
    return coeffs.sum(axis=0)


def _atrous_conv_axis0(x: np.ndarray, step: int) -> np.ndarray:
    """
    Convolucion B3-spline a trous a lo largo del eje 0, vectorizada
    sobre los ejes espaciales restantes. Padding reflect (semantica
    numpy). Taps del kernel en offsets [-2,-1,0,1,2]*step.
    """
    h = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    offsets = np.array([-2, -1, 0, 1, 2]) * step
    pad = 2 * step
    pad_width = [(pad, pad)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad_width, mode="reflect")
    n = x.shape[0]
    out = np.zeros_like(x)
    for w, off in zip(h, offsets):
        out += w * xp[pad + off: pad + off + n]
    return out


# ==================================================================
# 3. Thresholding
# ==================================================================

def soft_threshold(x: np.ndarray, thr: float | np.ndarray) -> np.ndarray:
    """Soft threshold (proximal de L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def estimate_sigma_subband(coeff_subband: np.ndarray, s_factor: float = 1.0) -> float:
    """
    Estima sigma del ruido en una subbanda via MAD robusto.

    s_factor : tu factor de correccion de asimetria (~1.56-1.57) si
               aplica; corrige la inflacion del MAD por senal residual.
               Para una calibracion mas fiel en tu caso, reemplaza
               esta funcion por la propagacion del CUBO NEGATIVO a
               traves de la MISMA transformada 2D-1D y mide su MAD.
    """
    data = np.asarray(coeff_subband)
    # para coeficientes complejos, estima sigma por componente
    # (parte real e imaginaria comparten la misma sigma del ruido)
    if np.iscomplexobj(data):
        vals = np.concatenate([data.real.ravel(), data.imag.ravel()])
    else:
        vals = data.ravel()
    mad = np.median(np.abs(vals - np.median(vals)))
    sigma = 1.4826 * mad
    return sigma / s_factor


def estimate_global_sigma_from_negatives(cube: np.ndarray) -> float:
    """
    Sigma global del ruido a partir de la cola NEGATIVA del cubo.

    La emision es no-negativa, asi que los pixeles con valor < 0 son ruido
    puro (no contaminados por senal). Se reflejan a positivo para formar una
    muestra simetrica centrada en 0 y se estima sigma via MAD robusto. Esta
    es la version "cubo negativo" de tu metodologia, condensada a un escalar
    que luego se PROPAGA por la transformada 2D-1D para obtener el sigma por
    subbanda (ver calibrate_sigma_2d1d).
    """
    data = np.asarray(cube, dtype=float).ravel()
    neg = data[data < 0]
    if neg.size == 0:
        vals = data - np.median(data)
    else:
        vals = np.concatenate([neg, -neg])   # simetrizar la cola de ruido
    mad = np.median(np.abs(vals - np.median(vals)))
    return 1.4826 * mad


def calibrate_sigma_2d1d(shape, ct: "CurveletTransform2D", n_spec_scales: int,
                         sigma_global: float, n_real: int = 1, seed: int = 0):
    """
    Calibra sigma por (subbanda curvelet, escala espectral) PROPAGANDO una
    realizacion de RUIDO blanco por la MISMA transformada 2D-1D.

    Las curvelets (UDCT) son un tight frame muy redundante y la starlet 1D es
    no-decimada: el ruido se redistribuye/colorea distinto en cada subbanda,
    asi que un MAD ingenuo sobre los datos escala mal los umbrales. Aqui se
    genera ruido gaussiano N(0, sigma_global) con la forma del cubo, se pasa
    por forward_2d1d y se mide su sigma por subbanda -> umbrales bien escalados
    (Monte-Carlo de propagacion de ruido, estandar en denoising starlet).

    Devuelve sigma_map: lista (por subbanda b) de np.ndarray de shape
    (n_spec_scales+1,) con la sigma del ruido por escala espectral.
    """
    rng = np.random.default_rng(seed)
    acc = None
    n_real = max(1, int(n_real))
    for _ in range(n_real):
        noise = rng.normal(0.0, float(sigma_global), size=tuple(shape))
        coeffs = forward_2d1d(noise, ct, n_spec_scales)
        sig = [np.array([estimate_sigma_subband(w1d[s], s_factor=1.0)
                         for s in range(w1d.shape[0])])
               for w1d in coeffs]
        acc = sig if acc is None else [a + b for a, b in zip(acc, sig)]
    if n_real > 1:
        acc = [a / n_real for a in acc]
    return acc


# ==================================================================
# 4. Operador 2D-1D: forward / threshold / inverse
# ==================================================================
# Estructura de datos: tras el forward 2D-1D, los coeficientes son
#   coeffs[b] -> array de shape (n_spec_scales+1, n_chan, ny_b, nx_b)
# donde b indexa la subbanda curvelet (escala+orientacion espacial),
# el primer eje es la escala ESPECTRAL (wavelet 1D no-decimada) y los
# tres restantes son (canal, y, x) de esa subbanda espacial.

def forward_2d1d(cube: np.ndarray, ct: CurveletTransform2D, n_spec_scales: int):
    """
    cube : (n_chan, ny, nx)
    1) curvelet 2D por canal -> subbandas espaciales
    2) wavelet 1D no-decimada a lo largo del eje espectral, por subbanda
    """
    n_chan = cube.shape[0]

    # paso 1: curvelet de cada canal; apilar subbandas a lo largo de chan
    per_chan = [ct.forward(cube[c]) for c in range(n_chan)]
    n_sub = len(per_chan[0])

    coeffs_2d1d = []
    for b in range(n_sub):
        # apilar la subbanda b de todos los canales -> (n_chan, ny_b, nx_b)
        stack = np.stack([per_chan[c][b] for c in range(n_chan)], axis=0)
        # paso 2: wavelet 1D espectral (eje 0 = canal), vectorizada
        # -> (n_spec_scales+1, n_chan, ny_b, nx_b)
        coeffs_2d1d.append(starlet_1d_forward(stack, n_spec_scales))
    return coeffs_2d1d


def inverse_2d1d(coeffs_2d1d, ct: CurveletTransform2D, n_chan: int):
    """Inversa exacta del forward_2d1d."""
    n_sub = len(coeffs_2d1d)
    # reconstruir cada subbanda espacial colapsando el eje espectral
    per_chan_subbands = [[None] * n_sub for _ in range(n_chan)]
    for b in range(n_sub):
        w1d = coeffs_2d1d[b]                       # (ns+1, n_chan, ny_b, nx_b)
        rec = starlet_1d_inverse(w1d)              # (n_chan, ny_b, nx_b)
        for c in range(n_chan):
            per_chan_subbands[c][b] = rec[c]
    # curvelet inverse por canal
    cube = np.stack([ct.inverse(per_chan_subbands[c]) for c in range(n_chan)],
                    axis=0)
    return cube


def threshold_2d1d(coeffs_2d1d, k_sigma: float = 3.0, s_factor: float = 1.0,
                   sigma_map=None, threshold_approx: bool = True,
                   protect_lowpass: bool = True):
    """
    Soft-threshold por subbanda curvelet. sigma se estima por subbanda
    (las curvelets son tight frame redundante: el ruido se reparte
    distinto entre subbandas, por eso NO un umbral global). Si se pasa
    `sigma_map` (calibrado por propagacion de ruido), se usa en lugar del
    MAD sobre los datos.

    El plano de APROXIMACION espectral (ultimo indice del eje 0) SI se
    umbraliza ESPACIALMENTE cuando threshold_approx=True: ese plano es de
    baja frecuencia solo en el eje espectral, pero conserva toda la
    frecuencia ESPACIAL (incluido el ruido espacial), asi que dejarlo pasar
    intacto reinyecta el ruido en la reconstruccion. Solo se protege el
    lowpass ESPACIAL (subbanda curvelet b==0) si protect_lowpass=True, para
    conservar el continuo / emision extendida.
    """
    out = []
    for b, w1d in enumerate(coeffs_2d1d):
        w_new = w1d.copy()
        n_spec = w1d.shape[0]
        s_max = n_spec if threshold_approx else n_spec - 1
        for s in range(s_max):
            is_spec_approx = (s == n_spec - 1)
            if is_spec_approx and protect_lowpass and b == 0:
                continue                                  # protege continuo
            sub = w1d[s]
            if sigma_map is not None:
                sigma = sigma_map[b][s]
            else:
                sigma = estimate_sigma_subband(sub, s_factor=s_factor)
            w_new[s] = soft_threshold(sub, k_sigma * sigma)
        out.append(w_new)
    return out


def threshold_2d1d_reweighted(
    coeffs_data,
    coeffs_model=None,
    k_sigma: float = 3.0,
    s_factor: float = 1.0,
    eps: float = 1e-33,
    first_pass: bool = False,
    sigma_map=None,
    threshold_approx: bool = True,
    protect_lowpass: bool = True,
):
    """
    Soft-threshold REWEIGHTED L1 (Candes, Wakin & Boyd 2008) por subbanda
    curvelet, adaptado del bucle de re-weighting de Denoiser2D1D.

    Siempre se umbralizan los coeficientes de los DATOS (`coeffs_data`),
    cuyo umbral base por (subbanda, escala espectral) es
        base = k_sigma * sigma_datos.
    La "memoria" del esquema vive en los PESOS, que se recalculan en cada
    pasada a partir de la descomposicion del MODELO previo (`coeffs_model`):

        w = (k_sigma * sigma_modelo) / (|coef_modelo| + eps)

    de modo que los coeficientes grandes (senal fuerte) reciben peso ~0
    -> umbral ~0 -> casi sin sesgo de encogimiento; los pequenos reciben
    peso grande -> mas sparsity. El peso solo se aplica a coeficientes
    SIGNIFICATIVOS del dato (|dato| > base); el resto usa el umbral base.

    first_pass=True (it=0): equivale a un soft-threshold plano (pesos=1),
    por lo que `coeffs_model` se ignora.

    El plano de APROXIMACION espectral (ultimo indice del eje 0) SI se
    umbraliza ESPACIALMENTE cuando threshold_approx=True (ver nota en
    threshold_2d1d): solo se protege el lowpass ESPACIAL (b==0) si
    protect_lowpass=True. Si se pasa `sigma_map` (calibrado por propagacion
    de ruido) se usa en lugar del MAD sobre los datos.
    """
    out = []
    for b in range(len(coeffs_data)):
        w1d_data = coeffs_data[b]
        w_new = w1d_data.copy()
        n_spec = w1d_data.shape[0]
        s_max = n_spec if threshold_approx else n_spec - 1
        for s in range(s_max):
            is_spec_approx = (s == n_spec - 1)
            if is_spec_approx and protect_lowpass and b == 0:
                continue                                  # protege continuo
            data_sub = w1d_data[s]
            if sigma_map is not None:
                sigma_data = sigma_map[b][s]
            else:
                sigma_data = estimate_sigma_subband(data_sub, s_factor=s_factor)
            base_thr = k_sigma * sigma_data

            if first_pass or coeffs_model is None:
                thr = base_thr
            else:
                model_sub = coeffs_model[b][s]
                if sigma_map is not None:
                    sigma_model = sigma_map[b][s]
                else:
                    sigma_model = estimate_sigma_subband(model_sub, s_factor=s_factor)
                weights = (k_sigma * sigma_model) / (np.abs(model_sub) + eps)
                # peso adaptativo solo en coeficientes significativos del dato
                mask = np.abs(data_sub) > base_thr
                thr = np.where(mask, weights * base_thr, base_thr)

            w_new[s] = soft_threshold(data_sub, thr)
        out.append(w_new)
    return out


# ==================================================================
# 5. Envoltura IST (Iterative Soft Thresholding)
# ==================================================================
# Estilo Lahiry et al. 2026: en la practica reweighted L1 con criterio
# de convergencia por PLATEAU de la varianza residual (no numero fijo
# de iteraciones). Aqui IST clasico con parada por plateau; el
# enganche para reweighting esta marcado.

def ist_denoise_cube(
    cube: np.ndarray,
    n_spec_scales: int = 4,
    k_sigma: float = 3.0,
    s_factor: float = 1.0,
    max_iter: int = 50,
    plateau_tol: float = 1e-4,
    plateau_patience: int = 3,
    ct_kwargs: dict | None = None,
    verbose: bool = True,
):
    """
    Denoising 2D-1D curvelet(x)wavelet con IST.

    Restriccion de positividad: el flujo de emision es no-negativo, por lo
    que se proyecta el modelo a >= 0 tras cada actualizacion (estilo
    Denoiser2D1D de wavelet_denoising.py).

    Parada: cuando la varianza del residual deja de bajar mas que
    plateau_tol (relativo) durante plateau_patience iteraciones.

    Devuelve el MEJOR modelo (el de menor varianza residual a lo largo de
    las iteraciones), no el ultimo, para no degradar tras el plateau.
    """
    ct_kwargs = ct_kwargs or {}
    n_chan, ny, nx = cube.shape
    ct = CurveletTransform2D(shape=(ny, nx), **ct_kwargs)

    estimate = np.zeros_like(cube)
    prev_resid_var = np.inf
    stall = 0

    # seguimiento del mejor modelo (menor varianza residual)
    best_estimate = estimate.copy()
    best_resid_var = np.inf
    best_iter = -1

    for it in range(max_iter):
        residual = cube - estimate
        coeffs = forward_2d1d(residual, ct, n_spec_scales)
        coeffs_t = threshold_2d1d(coeffs, k_sigma=k_sigma, s_factor=s_factor)
        # --- ENGANCHE REWEIGHTED L1 (Candes, Wakin & Boyd 2008) ---
        # recalcular pesos w = 1/(|coeff| + eps) y aplicarlos al umbral
        # en la siguiente pasada para aproximar L0.
        update = inverse_2d1d(coeffs_t, ct, n_chan)
        estimate = estimate + update
        # restriccion de positividad (flujo de emision no-negativo)
        estimate = np.maximum(0.0, estimate)

        resid_var = float(np.var(cube - estimate))
        rel_change = abs(prev_resid_var - resid_var) / (prev_resid_var + 1e-30)

        # actualizar mejor modelo
        if resid_var < best_resid_var:
            best_resid_var = resid_var
            best_estimate = estimate.copy()
            best_iter = it

        if verbose:
            print(f"iter {it:2d} | resid_var={resid_var:.6e} "
                  f"| rel_change={rel_change:.2e}")
        if rel_change < plateau_tol:
            stall += 1
            if stall >= plateau_patience:
                if verbose:
                    print(f"Plateau alcanzado en iter {it}.")
                break
        else:
            stall = 0
        prev_resid_var = resid_var

    if verbose:
        print(f"Mejor modelo: iter {best_iter} "
              f"(resid_var={best_resid_var:.6e}).")
    return best_estimate


# ==================================================================
# 6. Envoltura REWEIGHTED L1 (recomendada)
# ==================================================================
# Analoga al bucle de re-weighting de Denoiser2D1D._denoise_iterative_soft
# (Candes, Wakin & Boyd 2008), adaptada a la transformada curvelet 2D-1D.
#
# Diferencia clave frente a ist_denoise_cube:
#   - ist_denoise_cube: IST aditivo sobre el RESIDUAL (estimate += inv(thr(res))).
#   - aqui: en cada iteracion se re-umbralizan SIEMPRE los coeficientes de
#     los DATOS, y la memoria del proceso vive en los PESOS calculados a
#     partir del modelo previo. Esto reproduce el "model_update = datos"
#     del original (donde el paso de gradiente con mu=0.5 se cancela) y
#     aproxima L0 reduciendo el sesgo del soft-threshold sobre la senal
#     fuerte.

def reweighted_l1_denoise_cube(
    cube: np.ndarray,
    n_spec_scales: int = 4,
    k_sigma: float = 3.0,
    s_factor: float = 1.0,
    max_iter: int = 20,
    plateau_tol: float = 1e-4,
    plateau_patience: int = 3,
    eps: float = 1e-33,
    ct_kwargs: dict | None = None,
    verbose: bool = True,
    reference: np.ndarray | None = None,
    noise_cube: np.ndarray | None = None,
    sigma_global: float | None = None,
    calibrate: bool = True,
    calib_realizations: int = 1,
    threshold_approx: bool = True,
    protect_lowpass: bool = True,
    seed: int = 0,
):
    """
    Denoising 2D-1D curvelet(x)wavelet con REWEIGHTED L1.

    En cada iteracion:
      1) se umbralizan los coeficientes (fijos) de los DATOS con pesos
         adaptativos derivados del modelo previo (threshold_2d1d_reweighted),
      2) se reconstruye y se proyecta a positividad (flujo no-negativo),
      3) se actualizan pesos para la siguiente pasada.

    Mejoras frente a la version base:
      (#1) Se umbraliza tambien el plano de APROXIMACION espectral de forma
           espacial (threshold_approx=True), protegiendo solo el lowpass
           espacial (protect_lowpass=True). Antes ese plano se dejaba pasar
           intacto y reinyectaba el ruido espacial -> apenas habia denoising.
      (#2) CALIBRACION del ruido por propagacion: se estima sigma_global de la
           cola negativa del cubo (o se usa `sigma_global`/`noise_cube`) y se
           propaga por la MISMA transformada 2D-1D para obtener sigma por
           subbanda (calibrate=True). Escala bien los umbrales en el frame
           redundante curvelet+starlet.
      (#3) SELECCION del mejor modelo: si se pasa `reference` (cubo limpio),
           se elige el de menor RMSE vs referencia (criterio honesto). Sin
           referencia se devuelve el modelo CONVERGIDO (no el de menor
           varianza residual, que estaria sesgado a no denoisar).

    Parametros nuevos
    -----------------
    reference : cubo limpio para seleccionar el mejor modelo por RMSE.
    noise_cube : realizacion de ruido a propagar (si se da, manda sobre todo).
    sigma_global : sigma del ruido a propagar (si None y calibrate, se estima
                   de la cola negativa del cubo).
    calibrate : activa la calibracion por propagacion de ruido (#2).
    calib_realizations : nº de realizaciones Monte-Carlo para promediar sigma.
    threshold_approx, protect_lowpass : ver threshold_2d1d_reweighted (#1).
    """
    ct_kwargs = ct_kwargs or {}
    n_chan, ny, nx = cube.shape
    ct = CurveletTransform2D(shape=(ny, nx), **ct_kwargs)

    # coeficientes de los datos: constantes a lo largo de las iteraciones
    # (el paso de gradiente del esquema original reduce model_update a los
    # datos), solo cambian los pesos derivados del modelo.
    coeffs_data = forward_2d1d(cube, ct, n_spec_scales)

    # --- (#2) calibracion del ruido por propagacion 2D-1D ---
    sigma_map = None
    if calibrate:
        if noise_cube is not None:
            cn = forward_2d1d(noise_cube, ct, n_spec_scales)
            sigma_map = [np.array([estimate_sigma_subband(w1d[s], s_factor=1.0)
                                   for s in range(w1d.shape[0])])
                         for w1d in cn]
        else:
            if sigma_global is None:
                sigma_global = estimate_global_sigma_from_negatives(cube)
            sigma_map = calibrate_sigma_2d1d(
                (n_chan, ny, nx), ct, n_spec_scales, sigma_global,
                n_real=calib_realizations, seed=seed,
            )
        if verbose:
            print(f"(*) Calibracion de ruido: sigma_global="
                  f"{sigma_global if sigma_global is not None else float('nan'):.4e}"
                  f" | {len(sigma_map)} subbandas calibradas.")

    use_reference = reference is not None

    model = np.maximum(0.0, cube)
    prev_resid_var = np.inf
    stall = 0

    best_model = model.copy()
    best_score = np.inf
    best_iter = -1

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    for it in range(max_iter):
        first_pass = (it == 0)
        # pesos a partir del modelo previo (no se necesitan en la 1a pasada)
        coeffs_model = None if first_pass else forward_2d1d(model, ct, n_spec_scales)

        coeffs_t = threshold_2d1d_reweighted(
            coeffs_data, coeffs_model,
            k_sigma=k_sigma, s_factor=s_factor, eps=eps, first_pass=first_pass,
            sigma_map=sigma_map, threshold_approx=threshold_approx,
            protect_lowpass=protect_lowpass,
        )
        model = inverse_2d1d(coeffs_t, ct, n_chan)
        model = np.maximum(0.0, model)   # restriccion de positividad

        resid_var = float(np.var(cube - model))
        rel_change = abs(prev_resid_var - resid_var) / (prev_resid_var + 1e-30)

        # --- (#3) seleccion del mejor modelo ---
        if use_reference:
            score = _rmse(model, reference)
            if score < best_score:
                best_score = score
                best_model = model.copy()
                best_iter = it
        else:
            # sin referencia: nos quedamos con el modelo convergido (ultimo)
            best_model = model.copy()
            best_iter = it
            best_score = resid_var

        if verbose:
            extra = f" | rmse_ref={best_score:.6e}" if use_reference else ""
            print(f"iter {it:2d} | resid_var={resid_var:.6e} "
                  f"| rel_change={rel_change:.2e}{extra}")
        if rel_change < plateau_tol:
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
    return best_model


# ==================================================================
# NOTA DE CALIBRACION DE RUIDO (critico para tu caso)
# ==================================================================
# La estimacion via MAD por subbanda es un punto de partida. Para
# fidelidad con tu metodologia del cubo negativo:
#   1) pasar el CUBO NEGATIVO por forward_2d1d con la MISMA ct,
#   2) medir MAD de sus coeficientes por subbanda -> sigma empirico,
#   3) usar ese sigma (con tu factor s ~1.56-1.57) en threshold_2d1d.
# Esto respeta que la curvelet redistribuye el ruido correlacionado
# de forma distinta a la wavelet, y que el clip/denoising blanquea
# parcialmente el ruido de forma no uniforme.

if __name__ == "__main__":
    print(__doc__)
    print("Backend conectado: `curvelets` (UDCT). "
          "Recomendado: reweighted_l1_denoise_cube(cube) sobre un cubo "
          "(n_chan, ny, nx). Alternativa: ist_denoise_cube(cube).")