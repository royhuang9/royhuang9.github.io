# How to Access Jupyter notebook remotely
For example, access Jupyter notebook Google Cloud or AWS from your browser

All following operations are executed on remote machine or cloud
## Generate a password for jupyter notebook
```
$ jupyter notebook password
```
Enter your password

## Generate jupyter notebook configuration file
If you don’t already have config file for the notebook, create one using the following command:
```
$ jupyter notebook --generate-config
```
## Endit configuration file
```
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 9999
```

### Change the firewall to allow access tcp port 9999
Google it.

