# Final Project

Clone repo to location.
Navigate to that location.

How to build the docker container:

`docker build -t mikeditri/class9:latest .`

How to run the docker container:

inlcude `winpty` if running on windows machine

`winpty docker run -ti -v /${pwd.local_where_the_files_are}:/app mikeditri/class9:latest`


