# DDQN-PER-Dueling

## Dependencies
The implementation use the following Python modules:
```
tensorflow == 1.15
keras == 1.1.2
gym == 0.17.2
atary-py == 0.2.6
```
CUDA 10 is also needed for running on GPUs. The main.py le receive the following parameter:

## Main parameters
--game = atari game name
--mode = Training or Testing
--testing_path = path in which model are saved
--PER = boolean, using or not Prioritized experience replay
--dueling = boolean, using or not Dueling architecture
--render = render or not the game, in testing mode it is always rendered and scaled up

After the testing of a network a gif of the game with the highest score is created. 

## Running
An example of training command:
```
py main.py --mode Training --render --dueling
```
An example of testing command:
```
py main.py --mode Testing --PER --testing_path model/
```
## Installing
For the installing is sufficient to open a python virtual environment and execute the following:
```
pip install -r requirements.txt
```
Lastly, the file readlog.py is needed to read and eventually compare the log le written during
the training of the networks.

## References
- Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
- Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8.3-4 (1992):279-292.
- Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
- Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Thirtieth AAAI conference on articial intelligence. 2016.
- Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." Inter-national conference on machine learning. 2016.
- Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
