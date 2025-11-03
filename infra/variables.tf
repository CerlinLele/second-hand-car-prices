# Terraform变量定义文件
# 这个文件定义了所有可配置的参数

# AWS基础配置
variable "aws_region" {
  description = "AWS区域，例如：us-west-2, ap-northeast-1"
  type        = string
  default     = "us-east-1"  # 默认使用美国西部2区（俄勒冈）
}

variable "project_name" {
  description = "项目名称，用于资源命名"
  type        = string
  default     = "car-price-prediction"
  
  # 验证项目名称格式
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "项目名称只能包含小写字母、数字和连字符。"
  }
}

variable "environment" {
  description = "环境名称：开发(dev)、测试(staging)、生产(prod)"
  type        = string
  default     = "dev"
  
  # 限制环境名称的可选值
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "环境必须是以下之一：dev, staging, prod。"
  }
}
