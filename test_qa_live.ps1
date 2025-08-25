# QA Engine Test Script
Write-Host "🔥 GEOTHERMAL QA ENGINE - LIVE TEST" -ForegroundColor Red
Write-Host "=================================" -ForegroundColor Red
Write-Host ""

# Wait for server to start
Write-Host "⏳ Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Test 1: Health Check
Write-Host "1. 🏥 Health Check:" -ForegroundColor Green
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 10
    Write-Host "   ✅ Server is healthy!" -ForegroundColor Green
    $health | ConvertTo-Json -Depth 3
} catch {
    Write-Host "   ❌ Server not responding: $_" -ForegroundColor Red
    Write-Host "   💡 Make sure the server is running with: python -m uvicorn src.api:app --reload" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 2: Available Fields
Write-Host "2. 📋 Available Fields:" -ForegroundColor Green
try {
    $fields = Invoke-RestMethod -Uri "http://127.0.0.1:8000/fields" -Method Get -TimeoutSec 10
    Write-Host "   ✅ Found fields:" -ForegroundColor Green
    $fields.fields -join ", "
} catch {
    Write-Host "   ⚠️ Could not get fields: $_" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Ask Questions
$questions = @(
    @{question="Where is the caprock and how thick is it?"; field="Semurup"},
    @{question="What is the reservoir temperature?"; field=$null},
    @{question="How does MT data show subsurface structure?"; field=$null},
    @{question="Explain caprock lithologies in volcanic settings"; field=$null}
)

$i = 3
foreach ($q in $questions) {
    Write-Host "$i. 🤖 Question: $($q.question)" -ForegroundColor Green
    if ($q.field) {
        Write-Host "   🎯 Field: $($q.field)" -ForegroundColor Cyan
    }
    
    try {
        $body = @{question = $q.question}
        if ($q.field) { $body.field = $q.field }
        
        $jsonBody = $body | ConvertTo-Json -Compress
        Write-Host "   ⏳ Asking..." -ForegroundColor Yellow
        
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body $jsonBody -TimeoutSec 30
        
        Write-Host "   ✅ Response received!" -ForegroundColor Green
        Write-Host "   🎯 Intent: $($response.intent)" -ForegroundColor Cyan
        Write-Host "   📊 Confidence: $($response.confidence)" -ForegroundColor Cyan
        Write-Host "   📚 Evidence: $($response.text_chunks_found) text chunks, $($response.figures_found) figures" -ForegroundColor Cyan
        
        if ($response.citations.Count -gt 0) {
            Write-Host "   📖 Citations: $($response.citations -join ', ')" -ForegroundColor Cyan
        }
        
        Write-Host "   💬 Answer:" -ForegroundColor Yellow
        $answer = $response.answer
        if ($answer.Length -gt 200) {
            $answer = $answer.Substring(0, 200) + "..."
        }
        Write-Host "      $answer" -ForegroundColor White
        
    } catch {
        Write-Host "   ❌ Error: $_" -ForegroundColor Red
    }
    
    Write-Host ""
    $i++
}

Write-Host "🎉 Testing complete!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 To test more questions, use:" -ForegroundColor Yellow
Write-Host "   Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' -Method Post -ContentType 'application/json' -Body '{\"question\":\"Your question here\"}'"
Write-Host ""
Write-Host "🌐 Or use the interactive docs at: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
