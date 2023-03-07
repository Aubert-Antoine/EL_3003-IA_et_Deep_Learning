# EL-3003-IA-et-Deep-Learning

https://www.laurentnajman.org/index.php?page=ia-et-deep-learning 


# Google-Colab remote connexion : 
#### Documentation : 
https://www.jetbrains.com/help/pycharm/running-ssh-terminal.html
https://www.jetbrains.com/help/pycharm/remote-development-starting-page.html
https://pypi.org/project/ssh-Colab/

### ssh agent : 
https://dashboard.ngrok.com/get-started/setup 



### In Colab 
[GitHub of Colab-ssh](https://github.com/WassimBenzarti/colab-ssh)

[pypi.org colab-ssh](https://pypi.org/project/colab-ssh/)
1. Launch a Colab notebook. Choose a runtime type you prefer.
2. In a new colab notebook cell : `!pip install colab_ssh --upgrade`
3. ```jupyter
   from colab_ssh import launch_ssh, init_git
   launch_ssh('2MMglfT6q5La7SA9Byf6WTSve3w_4THAh8Lqh2fr23Fyp2Zbt','passW1')
    ```
4. To disable ngrok tunnels created, run the command below:
   `sshColab.kill()`


### In Pycharm : 
Just select the ssh remote interrpreteur