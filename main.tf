provider "aws" {
    region = "us-west-2"
}

resource "aws_instance" "app_server" {
    ami = "ami-005e54dee72cc1d00"
    instance_type = "t2.micro"
    tags = {
        Name = "terraform-example"
    }
}