$templateFile = "experiments\2025_08_21\shear_dan_750.yaml"
$seeds = @(123, 456, 789,9999)
$startVersion = 2

# Zkontroluj, zda existuje template
if (-not (Test-Path $templateFile)) {
    Write-Host "Template $templateFile does not exist!" -ForegroundColor Red
    exit
}

# Ulož originální obsah
$originalContent = Get-Content $templateFile -Raw

# Najdi base name bez suffixu _vX (i pokud je suffix řetězený)
$nameMatch = [regex]::Match($originalContent, "(?m)^name:\s*(.+?)\s*$")
if (-not $nameMatch.Success) {
    Write-Host "Template $templateFile does not contain a valid 'name:' field!" -ForegroundColor Red
    exit
}

$baseName = $nameMatch.Groups[1].Value -replace "(_v\d+)+$", ""
$templateDir = Split-Path -Parent $templateFile
$templateStem = [System.IO.Path]::GetFileNameWithoutExtension($templateFile)
$tempFiles = @()

try {
    $seedCounter = $startVersion
    foreach ($seed in $seeds) {
        # Sestav config pro konkrétní seed a verzi
        $content = $originalContent
        $content = $content -replace "(?m)^seed:\s*\d+\s*$", "seed: $seed"
        $content = $content -replace "(?m)^name:\s*.+$", "name: ${baseName}_v${seedCounter}"

        # Ulož dočasný config soubor
        $tempFileName = ".tmp_${templateStem}_seed${seed}_v${seedCounter}_$([guid]::NewGuid().ToString('N')).yaml"
        $tempFile = Join-Path $templateDir $tempFileName
        $content | Set-Content $tempFile
        $tempFiles += $tempFile

        Write-Host "Training with seed: $seed (v${seedCounter})" -ForegroundColor Green
        python main.py $tempFile
        $exitCode = $LASTEXITCODE

        if (Test-Path $tempFile) {
            Remove-Item $tempFile -Force
        }

        if ($exitCode -ne 0) {
            throw "Training failed for seed $seed with exit code $exitCode"
        }

        $seedCounter++
    }
}
finally {
    foreach ($file in $tempFiles) {
        if (Test-Path $file) {
            Remove-Item $file -Force
        }
    }
    Write-Host "Temporary configs cleaned up" -ForegroundColor Cyan
    Write-Host "Template file was not modified" -ForegroundColor Cyan
}