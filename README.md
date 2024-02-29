#### Docker commands

Name of container registry: `gptserve`

Name of Repository: `chatgpt_gxp`

Name of WebApp: `gptgxp`

Name of resource group: `gptserve-rg`

Login command: `az acr login --name gptserve` or `docker login gptserve.azurecr.io`

The docker login command is used to log into a Docker registry. In your case, you're trying to log into an Azure Container Registry (ACR).

The username and password you need to use are the ones associated with the Azure Container Registry you're trying to log into.

You can retrieve these credentials from the Azure portal:

1. Navigate to your Azure Container Registry resource.
2. Click on "Access keys" in the left-hand menu.
3. Here you'll find the "Login server", "Username", and two passwords (password and password2). You can use either of the two passwords.


Alternatively, if you have the Azure CLI installed, you can retrieve the credentials using the following command:

`az acr credential show --name gptserve`

Replace gptserve with the name of your Azure Container Registry. This command will return the "username" and "passwords".

Once you have the username and password, you can use them to log in:

Replace yourusername and yourpassword with your actual username and password.

```
docker build -t gptserve.azurecr.io/chatgpt_gxp:v2 .
docker push gptserve.azurecr.io/chatgpt_gxp:v2
```

You can now either deploy the Docker container using the code below, or go to the Portal and manually update the tag version in the Deployment Center. 
```
az functionapp config container set 
--name gptgxp 
--resource-group gptserve-rg 
--docker-custom-image-name gptserve.azurecr.io/chatgpt_gxp:v2
--docker-registry-server-url https://gptserve.azurecr.io --docker-registry-server-user gptserve --docker-registry-server-password <PASSWORD>
```

Note: to get the password for the ACR login above use `az acr credential show --name` 

To run the Docker image locally: `docker run -p 8080:80 gptserve.azurecr.io/chatgpt_gxp:v2`