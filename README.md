# DDQN-PER-Dueling

The implementation use the following Python modules:

tensorflow == 1.15
keras == 1.1.2
gym == 0.17.2
atary-py == 0.2.6

CUDA 10 is also needed for running on GPUs. The main.py le receive the following parameter:

--game = atari game name
--mode = Training or Testing
--testing_path = path in which model are saved
--PER = boolean, using or not Prioritized experience replay
--dueling = boolean, using or not Dueling architecture
--render = render or not the game, in testing mode it is always rendered and scaled up

After the testing of a network a gif of the game with the highest score is created. An example
of training command:

py main.py --mode Training --render --dueling

An example of testing command:

py main.py --mode Testing --PER --testing_path model/

For the installing is sufficient to open a python virtual environment and execute the following:

pip install -r requirements.txt

Lastly, the file readlog.py is needed to read and eventually compare the log le written during
the training of the networks.
