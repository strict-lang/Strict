# Test Strict.Runner with the example calculator

Write-Host "Building Strict.Runner..." -ForegroundColor Cyan
dotnet build Strict.Runner\Strict.Runner.csproj

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nRunning example..." -ForegroundColor Cyan
    dotnet run --project Strict.Runner -- Strict.Runner\Examples\SimpleCalculator.strict
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}
