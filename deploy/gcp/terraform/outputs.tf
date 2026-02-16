output "api_url" {
  description = "URL of the deployed API service"
  value       = google_cloud_run_service.adcamp_api.status[0].url
}

output "dashboard_url" {
  description = "URL of the deployed dashboard service"
  value       = google_cloud_run_service.adcamp_dashboard.status[0].url
}

output "api_service_name" {
  description = "Name of the API Cloud Run service"
  value       = google_cloud_run_service.adcamp_api.name
}

output "dashboard_service_name" {
  description = "Name of the dashboard Cloud Run service"
  value       = google_cloud_run_service.adcamp_dashboard.name
}

output "secret_id" {
  description = "Secret Manager secret ID for API key"
  value       = google_secret_manager_secret.ark_api_key.secret_id
}

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "Deployment region"
  value       = var.region
}
