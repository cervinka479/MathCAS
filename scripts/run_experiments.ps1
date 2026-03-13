$templateFile = "experiments\2026_02_01\shear_dan_1500.yaml"
$seeds = @(456, 789, 9999)
$startVersion = 3

# Zkontroluj, zda existuje template
if (-not (Test-Path $templateFile)) {
    Write-Host "Template $templateFile neexistuje!" -ForegroundColor Red
    exit
}

# Ulož originální obsah
$originalContent = Get-Content $templateFile

$seedCounter = $startVersion
foreach ($seed in $seeds) {
    # Přečti originální YAML
    $content = Get-Content $templateFile
    
    # Nahraď seed
    $content = $content -replace "seed: \d+", "seed: $seed"
    
    # Nahraď name - přidej verzí na konec
    $content = $content -replace "name: (.+)", "name: `${1}_v${seedCounter}"
    
    # Ulož změny do template souboru
    $content | Set-Content $templateFile

    Write-Host "Training with seed: $seed (v${seedCounter})" -ForegroundColor Green
    python main.py $templateFile
    
    $seedCounter++
}

# Vrať originální obsah zpět
$originalContent | Set-Content $templateFile
Write-Host "Template obnoven na originální stav" -ForegroundColor Cyan