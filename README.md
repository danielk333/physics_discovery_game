# Physics modeling and discovery game

## Installing on student computers

1. Clone this repository
2. Run `pip install -r requirement.txt`
3. run `python compile.py "path/to/game"`. The path should NOT be inside the repository. 
4. Delete the repository.
5. Run the game with `python run.py [args]` from the `path/to/game` directory. All logfiles will now be stored in this directory.

After this, there should be a compiled game package in the given path, together with a `run.py` where the student input is stored. When this file is run with `python run.py` help text is displayed with available game commands. 

When the game is setup in an actual classroom, there should also be a prepared document with instructions in this folder specific for the course in question.

When deployed in a course, an additional layer of encryption should probably be put on the game config and passcodes file so that the game decrypts the files to read the data from them. This prevents to students from "cheating". This is currently not implemented by default so the files are visible in plain-text.

TODO:

[ ] Automatic encryption of files that only the `pyc` files can decrypt.

## Playing the game

### Background

The game is about exploring physics and moving on to new domains. You have the possibility to launch a automated probe with thrusters into a new realm governed by physics we do not know. There is only a limited amount of launch attempts so one cannot be wasteful and just brute force it. Much like spacecraft today, simulation prior to excursion is extremely useful when one can only afford to try something once or twice.

### Game components

The game has 3 major modes:
- exp [realm number]: Experiment mode, launches an experiment into the new realm. The probe records all data it can, including its position, velocity, mass and thrust generated. The goal is to reach the next portal, otherwise the probe will be stranded and lost, although we always get the logfiles back.
- log [log number]: Log quicklook mode, plots the logfile from an experiment to view. This data will need to be more carefully analyzed but a first look is always good to have.
- sim [realm number]: Perform a simulation of an experiment. The simulation uses the model given by the player to predict the trajectory of the probe. If the physics model is correct, this simulation can be used to figure out how to create the automated thruster program that will get us to the goal!

### A typical game round

1. Clear the thruster control system so that it is completely turned off.
2. Launch an experiment with `python run.py exp`.
3. Watch the outcome, get familiar with the environment.
4. Plot the log with `python run.py log`, look at the forces that affect the probe.
5. Load the logfile into some analysis program written by the player, create an analytic model of the force, write it in the `def force(...):` function in `run.py`.
6. Run a simulation with the new force model using `python run.py sim`. If the trajectory looks like the logfile, you have discovered and modeled the physics!
7. Write a parameter controlled thruster program, run a simulation again using `python run.py sim`, but this time tune the parameters using the TK interface.
8. Find a thruster program that gets to the goal, put those parameters into the `get_control_params` function.
9. Launch a new experiment with `python run.py exp` and cross your fingers! Hopefully you win!


## The bank

The main reason for the "bank" is possible students that simply try to spam solutions. They might get them right and pass one level but, it wont work for the advanced levels and with only a few chances they cannot spam multiple levels. This forces the students to model the physics so they wont run out of tries. If they need more chances, its not more than appropriate that to get more tries they need to solve problems and hand in solutions! The problems can even be tailored to help students understand the physics in the game.


## Usage

If this is used as an education tool, please let me know at:
Daniel Kastinen (daniel.kastinen@irf.se)

Pull-requests and forks are highly appreciated. The point is to tailor the "game" to the course in question. 

Some ideas that might help:
- Introduce a wall game-object that one can quantum tunnel trough for QM-course
- Modify force to Newtonian gravity for solar system course
- Modify force to Kerr-metric black hole for GR course
- Add magnetic force and modify forces to electric, create EM-obstacle-course.
- Add variational friction for mechanics course.
- Implement elastic collisions for mechanics course. 
- Remove the "simulation" part and give the students all level information, force them to solve the optimization problem by hand for an analytics/basic mathematics course (e.g fastest route problem, path of least energy, ect). 
