output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "api_url" {
  description = "URL for the API service"
  value       = "http://${aws_lb.main.dns_name}"
}

output "dashboard_url" {
  description = "URL for the Dashboard service"
  value       = "http://${aws_lb.main.dns_name}/dashboard"
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "api_service_name" {
  description = "Name of the API ECS service"
  value       = aws_ecs_service.api.name
}

output "dashboard_service_name" {
  description = "Name of the Dashboard ECS service"
  value       = aws_ecs_service.dashboard.name
}

output "api_log_group" {
  description = "CloudWatch log group for API"
  value       = aws_cloudwatch_log_group.api.name
}

output "dashboard_log_group" {
  description = "CloudWatch log group for Dashboard"
  value       = aws_cloudwatch_log_group.dashboard.name
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}
