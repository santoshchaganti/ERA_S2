terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.92"
    }
  }
  required_version = ">= 1.2"
}

variable "gemini_api_key" {}
variable "aws_access_key" {}
variable "aws_secret_key" {}
variable "aws_region" {}
variable "key_pair_name" {}
variable "vpc_id" {}

provider "aws" {
    region = var.aws_region
    access_key = var.aws_access_key
    secret_key = var.aws_secret_key
}

resource "tls_private_key" "rsa_4096" {
    algorithm = "RSA"
    rsa_bits  = 4096
}

resource "aws_key_pair" "key_pair" {
  key_name   = var.key_pair_name
  public_key = tls_private_key.rsa_4096.public_key_openssh
}

resource "local_file" "private_key" {
    content = tls_private_key.rsa_4096.private_key_pem
    filename = "${aws_key_pair.key_pair.key_name}.pem"
}

resource "aws_instance" "s2_module" {
  ami           = "ami-0bbdd8c17ed981ef9"
  instance_type = "t3.micro"
  key_name      = aws_key_pair.key_pair.key_name
  vpc_security_group_ids = [aws_security_group.s2_module_sg.id]
  tags = {
    Name = "s2_module"
  }
  user_data = <<-EOF
    #!/bin/bash
    apt-get update && apt-get upgrade -y
    snap install astral-uv --classic
    mkdir -p /home/ubuntu/projects && cd /home/ubuntu/projects
    git clone https://github.com/santoshchaganti/ERA_S2.git
    chown -R ubuntu:ubuntu /home/ubuntu/projects
    cd /home/ubuntu/projects/ERA_S2
    sudo -u ubuntu uv sync
    cat << 'ENV_EOF' > .env
    GEMINI_API_KEY = ${var.gemini_api_key}
    ENV_EOF
    chown ubuntu:ubuntu .env
    chmod 600 .env
    sudo -u ubuntu uv run main.py
  EOF
}

resource "aws_security_group" "s2_module_sg" {
  name        = "s2_module_sg"
  description = "s2_module_sg"
  vpc_id      = var.vpc_id
  tags = {
    Name = "s2_module_sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "s2_module_sg_http" {
  security_group_id = aws_security_group.s2_module_sg.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 80
  ip_protocol = "tcp"
  to_port     = 80
}

resource "aws_vpc_security_group_ingress_rule" "s2_module_sg_ssh" {
  security_group_id = aws_security_group.s2_module_sg.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 22
  ip_protocol = "tcp"
  to_port     = 22
}

resource "aws_vpc_security_group_ingress_rule" "s2_module_sg_https" {
  security_group_id = aws_security_group.s2_module_sg.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 443
  ip_protocol = "tcp"
  to_port     = 443
}

resource "aws_vpc_security_group_ingress_rule" "s2_module_sg_5000" {
  security_group_id = aws_security_group.s2_module_sg.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 5000
  ip_protocol = "tcp"
  to_port     = 5000
}

resource "aws_vpc_security_group_egress_rule" "s2_module_all_outbound" {
  security_group_id = aws_security_group.s2_module_sg.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}