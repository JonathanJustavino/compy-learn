# Docker test environment
**purpose**: getting working runtime environment without manual installation steps for:
 - checking setup files are working
 - have a reference environment when using on native environments

## Docker Scripts
| script         | description                                                                             |
|----------------|-----------------------------------------------------------------------------------------|
| `d-shell.sh`   | starting shell in docker environment, builds image and starts container if not yet done |
| `d-down.sh`    | stops running docker environment (removes running container)                            |
| `d-rebuild.sh` | rebuild docker image - run setup scripts again to build runtim environment              |

## Docker specific Files & Folders
```text
compy-learn
+- .docker-support
|  +- restore-binaries.sh  copy saved build assets to "mounted" directories
|  +- run-with-venv.sh     run command with active python virtual environment
|  +- save-binaries.sh     save build assets to separate folder to be available before not visible by "mounting" host folders
|  +- start-shell.sh       start shell in docker (add common setup steps)
+- .dockerignore           configuration of set of repo files not send to docker
+- d-down.sh               script: stop running docker container
+- d-rebuild.sh            script: trigger docker image rebuild
+- d-shell.sh              script: start shell in running docker container (starts container if not running)
+- DOCKER.md               this documentation file
+- docker-compose.yml      service configuration to run docker image
+- Dockerfile              build file to create docker image
```
