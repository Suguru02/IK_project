$pythonPath = "C:\ProgramData\miniconda3\envs\robotics\python.exe"
$scriptName = "graph_gen.py"
$outDir = "panda_dataset/tmp_chunks"
$chunkSize = 20
$totalSamples = 122000
$parallel = 5

# Создаём выходную папку, если её нет
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

for ($start = 120000; $start -lt $totalSamples; $start += $chunkSize * $parallel) {
    $jobs = @()
    for ($i = 0; $i -lt $parallel; $i++) {
        $s = $start + $i * $chunkSize
        $e = $s + $chunkSize - 1
        if ($s -ge $totalSamples) { break }
        $outFile = "$outDir/chunk_$s-$e.pt"
        # 👇 Внимание: передаём сначала имя скрипта, а потом аргументы к нему
        $argsList = "$scriptName --start $s --count $chunkSize --out_dir $outFile"
        Write-Host "Запуск: start=$s, count=$chunkSize"
        $job = Start-Process -NoNewWindow -FilePath $pythonPath -ArgumentList $argsList -PassThru
        $jobs += $job
    }
    # Ждём завершения этой партии
    $jobs | Wait-Process
    Write-Host "Партия до $($start + $chunkSize * $parallel - 1) завершена."
}