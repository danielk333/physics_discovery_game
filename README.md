# Physics modeling and discovery game

## Installing on student computers

1. Clone this repository
2. Run `pip install -r requirement.txt`
3. run `python3 compile.py "path/to/game/dir/not/in/repository"`
4. Delete the repository

After this, there should be a compiled game library in the given path, together with a `run.py` where the user input is stored. When this file is run `python3 run.py` a help with avalible game commands in shown. There should also be a PDF with game instructions in this folder.

## Playing the game

TODO: add description


## The bank

The main reason for the "bank" is that even if there are students that simply try to spam solutions, and happen to get them right, it wont work for the advanced levels. Then they will run out of tries and be forced to model the physics. But they need more chanses. And since these students ended up in that situation due to lack of modeling, its not more than appropirate that to get more tries they need to solve analytic problems and hand in solutions!


## Usage

If this is used as an education tool, please let me know at:
Daniel Kastinen (daniel.kastinen@irf.se)

Pull-requests and forks are highly apriciated. The point is to tailor the "game" to the course in question. 

Some ideas that might help:
- Introduce a wall game-object that one can quantum tunnel trough for QM-course
- Modify force to newtonian gravity for solar system course
- Modify force to Kerr-metric black hole for GR course
- Add magnetic force and modify forces to electric, create EM-obstacle-course.
- Add variational friction for mechanics course.
- Implement elastic collisions for mechanics course. 
- Remove the "simulation" part and give the students all level information, force them to solve the optimization problem by hand for an analytics/basic mathematics course (e.g fastest route problem, path of least energy, ect). 
