# Deploy Script for Windows (PowerShell)
$ErrorActionPreference = "Stop"

# --- CONFIGURATION ---
$PROJECT_ID = "my-project-id-45"   # <--- CONFIRM THIS MATCHES YOUR CONSOLE
$REGION = "us-central1"
$SERVICE_NAME = "antigravity-layer1-detector"
$GATEWAY_NAME = "antigravity-layer1-gateway"
$SECRET_NAME = "ANTIGRAVITY_API_KEY"
# ---------------------

Write-Host "Deploying to Project: $PROJECT_ID, Region: $REGION" -ForegroundColor Cyan

# Configure project
gcloud config set project $PROJECT_ID

$IMAGE = "gcr.io/$PROJECT_ID/$($SERVICE_NAME):v1"
$GATEWAY_IMAGE = "gcr.io/$PROJECT_ID/$($GATEWAY_NAME):v1"

# 1) Build Detector
Write-Host "Building Detector Image..." -ForegroundColor Green
gcloud builds submit --tag "$IMAGE" .
if ($LASTEXITCODE -ne 0) { throw "Detector build failed" }

# 2) Create Secret
Write-Host "Checking Secret..." -ForegroundColor Green
$secretExists = gcloud secrets describe "$SECRET_NAME" --project "$PROJECT_ID" 2>$null
if (-not $secretExists) {
  Write-Host "Creating secret $SECRET_NAME..."
  gcloud secrets create "$SECRET_NAME" --replication-policy="automatic" --project "$PROJECT_ID"
    
  $apiKey = Read-Host "Please enter the API Key value to store in Secret Manager"
  $apiKey | gcloud secrets versions add "$SECRET_NAME" --data-file=- --project "$PROJECT_ID"
}
else {
  Write-Host "Secret already exists. Using existing."
}

# 2.5) Gemini API Key
$GEMINI_KEY = $env:GEMINI_API_KEY
if (-not $GEMINI_KEY) {
  Write-Host "NOTE: You need a valid Google AI Studio Key." -ForegroundColor Yellow
  $GEMINI_KEY = Read-Host "Please enter your NEW Gemini API Key (starts with AIza...)"
}

# 3) Deploy Detector
Write-Host "Deploying Detector..." -ForegroundColor Green
gcloud run deploy "$SERVICE_NAME" `
  --image "$IMAGE" `
  --region "$REGION" `
  --project "$PROJECT_ID" `
  --set-secrets "ANTIGRAVITY_API_KEY=$SECRET_NAME`:latest" `
  --set-env-vars "GEMINI_API_KEY=$GEMINI_KEY" `
  --allow-unauthenticated `
  --memory "2Gi" `
  --concurrency 10
if ($LASTEXITCODE -ne 0) { throw "Detector deploy failed" }

# 4) Build Gateway
Write-Host "Building Gateway Image..." -ForegroundColor Green
gcloud builds submit gateway --tag "$GATEWAY_IMAGE" --project "$PROJECT_ID"
if ($LASTEXITCODE -ne 0) { throw "Gateway build failed" }

# Get Detector URL
$detectorUrl = gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --format='value(status.url)' --project=$PROJECT_ID
Write-Host "Detector URL: $detectorUrl"

# 5) Deploy Gateway
Write-Host "Deploying Gateway..." -ForegroundColor Green
gcloud run deploy "$GATEWAY_NAME" `
  --image "$GATEWAY_IMAGE" `
  --region "$REGION" `
  --project "$PROJECT_ID" `
  --set-env-vars "DETECTOR_URL=$detectorUrl" `
  --set-secrets "ANTIGRAVITY_API_KEY=$SECRET_NAME`:latest" `
  --allow-unauthenticated `
  --memory "1Gi" `
  --concurrency 50

Write-Host "Deployment Complete!" -ForegroundColor Cyan
$gatewayUrl = gcloud run services describe $GATEWAY_NAME --platform=managed --region=$REGION --format='value(status.url)' --project=$PROJECT_ID
Write-Host "Public Gateway URL: $gatewayUrl/api/chat" -ForegroundColor Yellow
