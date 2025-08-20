# Linode Instructions

### Steps:
1. **Install Terraform**: Ensure Terraform is installed on your system.
2. **Create Linode API Token**: Obtain a personal access token from Linode's cloud manager.
3. **Set up the Terraform configuration**: `linode.tf`

### List linodes

```sh
pip install linode-cli
linode-cli --version
```

```sh
linode-cli linodes list
```

```sh
linode-cli regions list
```

### Instructions:
### Set Linode Token
Export your Linode token as an environment variable or provide it as a `terraform.tfvars` file:
```hcl
linode_token = "xtoken"
```

### Init
```bash
terraform init
```

### Apply Configuration
```bash
terraform plan
terraform plan --out=planfile
```

```bash
terraform apply
```

### Targeting module

```sh
terraform plan -target=linode_instance.linode_servers
terraform apply -target=linode_instance.linode_servers
<service_name><instance_name>
```

### Destory

```bash
terraform destroy
```

- https://registry.terraform.io/providers/linode/linode/latest/docs
