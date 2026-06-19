filtros = {
    'F_MALLAT_7_9': 1, # Biortogonal 7/9 [POR DEFECTO]
    'F_DAUBE_4': 2, # Daubechies 4
    'F_BI2HAAR': 3, # Biortogonal 2 Haar
    'F_BI4HAAR': 4, # Biortogonal 4 Haar
    'F_ODEGARD_7_9': 5, # Odégard 7/9
    'F_5_3': 6, # 5/3
    'F_LEMARIE_1': 7, # Lemarie orden 1
    'F_LEMARIE_3': 8, # Lemarie orden 3
    'F_LEMARIE_5': 9, # Lemarie orden 5
    'F_USER': 10, # Custom (requiere archivo de filtro)
    'F_HAAR': 11, # Haar
    'F_3_5': 12, # 3/5
    'F_4_4': 13, # 4/4
    # F_5_3_DIV (14) y F_MALLAT_9_7 (15) existen en el enum pero el binding
    # los rechaza (validacion: value < NBR_SB_FILTER=14). Valores validos: 1-13
}

transformaciones_1d = {
    'T01_PAVE_LINEAR': 0, # A trous lineal no decimada
    'T01_PAVE_B1SPLINE': 1, # A trous B1-spline no decimada
    'T01_PAVE_B3SPLINE': 2, # A trous B3-spline no decimada, Starlet 1D
    'T01_PAVE_B3_DERIV': 3, # A trous B3-spline derivada
    'T01_PAVE_HAAR': 4, # Haar no decimada
    'TM1_PAVE_MEDIAN': 5, # Mediana multiresolución
    'TU1_MALLAT':6, # Mallat no decimada
    'TU1_UNDECIMATED_NON_ORTHO':7, # No ortogonal no decimada
    'T01_PAVE_B3SPLINE_GEN2':8, # B3-spline generación 2
    'T01_PYR_B3SPLINE':9, # Pirámide B3-spline decimada
    'TM1_PYR_MEDIAN':10, # Pirámide mediana
    'T01_PAVE_MORLET':11, # Wavelet de Morlet no decimada
    'T01_PAVE_MEX':12, # Mexican Hat no decimada
    'T01_PAVE_FRENCH':13, # French Hat no decimada
    'T01_PAVE_DERIV_GAUSS':14, # Derivada de Gaussiana no decimada
    'T01_MALLAT':15, # Mallat decimada [POR DEFECTO]
    'T01_LIFTING':16, # Lifting scheme
    'WP1_MALLAT':17, # Wavelet Packets Mallat
    'WP1_LIFTING':18, # Wavelet Packets Lifting
    'WP1_ATROUS':19, # Wavelet Packets A trous
    'T01_PYR_LINEAR':20 # Pirámide lineal
}

transformaciones_2d = {
    'T01_PAVE_LINEAR':1, # A trous con kernel lineal
    'TO_PAVE_BSPLINE':2, # A trous con B-spline (Starlet) [POR DEFECTO]
    'TO_PAVE_FFT':3, # A trous usando FFT
    'TM_PAVE_MEDIAN':4, # A trous con mediana
    'TM_PAVE_MINMAX':5, # A trous Min-Max
    'TO_PYR_LINEAR':6, # Pirámide con kernel lineal
    'TO_PYR_BSPLINE':7, # Pirámide con B-spline
    'TO_PYR_FFT_DIFF_RESOL':8, #Pirámide FFT con diferencia de resolución
    'TO_PYR_MEYER':9, # Pirámide con wavelet de Meyer
    'TM_PYR_MEDIAN':10, # Pirámide con mediana
    'TM_PYR_LAPLACIAN':11, # Pirámide Laplaciana
    'TM_PYR_MINMAX':12, # Pirámide Min-Max
    'TM_PYR_SCALING_FUNCTION':13, # Pirámide con función de escala
    'TO_MALLAT':14, # Transformada de Mallat (ortogonal no decimada)
    'TO_FEAUVEAU':15, # Transformada de Feauveau
    'TO_PAVE_FEAUVEAU':16, # Transformada de Feauveau no decimada
    'TO_LC':17, # Line-Column
    'TO_HAAR':18, # Haar
    'TO_SEMI_PYR':19, # Semi-pirámide
    'TM_TO_SEMI_PYR':20, # Semi-pirámide (morpho)
    'TO_DIADIC_MALLAT':21, # Mallat diádica
    'TM_TO_PYR':22, # Pirámide morfológica
    'TO_PAVE_HAAR':23, # Haar no decimada
    'TO_UNDECIMATED_MALLAT':24, # Mallat no decimada
    'TO_UNDECIMATED_NON_ORTHO':25, # No decimada No ortogonal
    'TO_PYR_MEYER_ISOTROP':26, # Mater isotrópica en pirámide
    'TO_PYR_FFT_DIFF_SQUARE':27, # Pirámide FFT con diferencia cuadrada
    'TC_FCT':28, # Fast Curvelet Transform
    'TO_LIFTING':29 # Lifting scheme
}