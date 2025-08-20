// Provider Configuration
provider "linode" {
  token = var.linode_token // Linode API token
}

// vars
variable "linode_token" {
  type      = string
  sensitive = true
}

variable "instance_type" {
  default     = "g1-gpu-rtx6000-1" // Dedicated GPU w/ RTX 6000 + 32GB RAM
  description = "Dedicated GPU (RTX 6000) instance type"
}

variable "region" {
  default     = "us-east" // Default region
  description = "US east"
}

variable "image" {
  default     = "linode/ubuntu24.04" // ubuntu24.04 image
  description = "linode/ubuntu24.04"
}

// Create Linode Instance
resource "linode_instance" "gpu_server" {
  label  = "gpu-rtx6000"
  region = var.region
  type   = var.instance_type
  image  = var.image
}

// Outputs
output "server_ip_address" {
  value       = linode_instance.gpu_server.ip_address
  description = "Public IP address of the GPU server"
}

output "server_label" {
  value       = linode_instance.gpu_server.label
  description = "Label of the GPU server"
}
