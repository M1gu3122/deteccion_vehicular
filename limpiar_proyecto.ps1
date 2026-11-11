param([switch]$Eliminar)

$ErrorActionPreference = 'Stop'
$raiz = [System.IO.Path]::GetFullPath($PSScriptRoot)
if (-not (Test-Path -LiteralPath (Join-Path $raiz 'dataset_autos_v2/train') -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $raiz 'dataset_autos_v2/val') -PathType Container)) {
    throw 'No se encontro el dataset de entrenamiento que se debe conservar.'
}
$registro = Get-Content -LiteralPath (Join-Path $raiz 'runs/auto_cls_v2/args.yaml') -Raw
if ($registro -notmatch 'data: [^\r\n]*dataset_autos_v2') {
    throw 'El registro de entrenamiento no confirma dataset_autos_v2.'
}
$sobrantes = @(
    'dataset_autos', 'dataset_autos_full',
    'yolo11n.pt', 'yolo11n-cls.pt', 'yolo11n_openvino_model',
    'modelos/auto_cls.pt', 'modelos/auto_cls_openvino_model',
    'runs/auto_cls', 'config.py.restaurar_bak', 'hoja_contacto.jpg',
    '__pycache__', 'entrenar_clasificador.py', 'preparar_dataset.py',
    'reorganizar_dataset.py', 'vitalizar_dataset.py',
    'diagnosticar_reentrenamiento.py', 'pruebas/test_clases.py',
    'pruebas/test_inferencia.py', 'pruebas/test_inferencia2.py'
)

foreach ($relativo in $sobrantes) {
    $ruta = [System.IO.Path]::GetFullPath((Join-Path $raiz $relativo))
    if (-not $ruta.StartsWith($raiz + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Ruta fuera del proyecto: $ruta"
    }
    if (-not (Test-Path -LiteralPath $ruta)) { continue }
    if ($Eliminar) {
        Remove-Item -LiteralPath $ruta -Recurse -Force
        Write-Output "Eliminado: $relativo"
    } else {
        Write-Output "Se eliminaria: $relativo"
    }
}

if (-not $Eliminar) {
    Write-Output 'Revision solamente. Ejecuta .\limpiar_proyecto.ps1 -Eliminar para aplicar.'
}
