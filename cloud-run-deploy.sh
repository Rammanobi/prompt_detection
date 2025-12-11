#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION (UPDATE THESE) ---
PROJECT_ID="my-project-id-45"    # <--- REPLACE THIS
REGION="us-central1"
SERVICE_NAME="antigravity-layer1-detector"
GATEWAY_NAME="antigravity-layer1-gateway"
SECRET_NAME="ANTIGRAVITY_API_KEY"
# ------------------------------------

echo "Deploying to Project: ${PROJECT_ID}, Region: ${REGION}"

IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:v1"
GATEWAY_IMAGE="gcr.io/${PROJECT_ID}/${GATEWAY_NAME}:v1"

# 1) Build and push Detector image
echo "Building Detector Image..."
gcloud builds submit --tag "${IMAGE}" .

# 2) Create Secret (if not exists)
if ! gcloud secrets describe "${SECRET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creating secret ${SECRET_NAME}..."
  # Create the secret wrapper
  gcloud secrets create "${SECRET_NAME}" --replication-policy="automatic" --project "${PROJECT_ID}"
  
  echo "Please enter the API Key value to store in Secret Manager:"
  read -s API_KEY_VAL
  
  # Add the version
  echo -n "$API_KEY_VAL" | gcloud secrets versions add "${SECRET_NAME}" --data-file=- --project "${PROJECT_ID}"
else
  echo "Secret ${SECRET_NAME} already exists. Using existing."
fi

# 3) Deploy detector (Private, access via internal network or specific IAM, but here we require header)
# Note: we disable unauthenticated for strict security, but Gateway needs permission to call it.
# For simplicity in this script, we might allow unauthenticated at network level but enforce KEY in app.
# However, best practice is --allow-unauthenticated=false and give Gateway Service Account 'run.invoker'.
# Let's clean up: We will enforce KEY in App. Network level:
echo "Deploying Detector..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --set-secrets "ANTIGRAVITY_API_KEY=${SECRET_NAME}:latest" \
  --allow-unauthenticated \
  --memory "1Gi" \
  --concurrency 10

# 4) Deploy Gateway
echo "Building Gateway Image..."
gcloud builds submit gateway --tag "${GATEWAY_IMAGE}" --project "${PROJECT_ID}"

# Get the detector URL
DETECTOR_URL=$(gcloud run services describe ${SERVICE_NAME} --platform=managed --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})
echo "Detector is running at: ${DETECTOR_URL}"

echo "Deploying Gateway..."
gcloud run deploy "${GATEWAY_NAME}" \
  --image "${GATEWAY_IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --set-env-vars "DETECTOR_URL=${DETECTOR_URL}" \
  --set-secrets "ANTIGRAVITY_API_KEY=${SECRET_NAME}:latest" \
  --allow-unauthenticated \
  --memory "512Mi" \
  --concurrency 50

echo "Deployment Complete."
echo "Public Gateway URL: $(gcloud run services describe ${GATEWAY_NAME} --platform=managed --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})/api/chat"
