#!/usr/bin/env bash

# sudo pip install virtualenv
# sudo pip install virtualenvwrapper

proj_name=jsk_hsr_wrs
venv=$HOME/.virtualenvs
var_name=WORKON_HOME

#! check if environment exists
if [ ! -d $venv ];
then
   /bin/echo -e "\e[1;32m[LOG] virtualenv does not exist. new one will be created\e[0m"
   
   if [ ! -d $venv ];
   then
       /bin/echo -e "\e[1;32m[LOG] creating virtualenv folder at-> "$venv"\e[0m"
       mkdir $venv
   fi
   
   if [[ -z "${WORKON_HOME}" ]];
   then
       printf '\n## Python Virtual Environment\n' >> $HOME/.bashrc
       echo "export "$var_name=$venv >> $HOME/.bashrc
       /bin/echo -e "\e[1;32m[LOG] setting env var ${WORKON_HOME}\e[0m"
   fi

   if [ ! -f /usr/local/bin/virtualenvwrapper.sh ];
   then
       /bin/echo -e "\e[1;31m[ERROR] File /usr/local/bin/virtualenvwrapper.sh not found\e[0m"
       exit 1
   fi
   
   source /usr/local/bin/virtualenvwrapper.sh
else
    source /usr/local/bin/virtualenvwrapper.sh
    /bin/echo -e "\e[1;32m[LOG] virtualenv exists\e[0m"
fi

if [ ! -d $WORKON_HOME/$proj_name ];
then
   /bin/echo -e "\e[1;32m[LOG] creating virtual environment for the project \e[0m"
   mkvirtualenv --python=python3 $proj_name
   workon $proj_name

   source $WORKON_HOME/$proj_name/bin/activate
else
    source $WORKON_HOME/$proj_name/bin/activate
fi


/bin/echo -e "\e[1;32m[LOG] installing dependencies\e[0m"
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pip install -r $dir/requirements.txt

