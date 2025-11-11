# Higher Dimensions

Higher Dimensions is a game project where the player will have to go **through different surfaces to discover their inner topology** to progress on the story.
Each level is a surface, with a given geometry (mountains, valleys, etc) and you have to discover what is the topology behind:

- _If I walk straight in one direction, will I get to the same point?_
- _Why is everything reversed if I make a loop to the north but not to the west?_
- _Am I on the surface of a sphere or is it a donut?_
- _What are the closed paths that I can walk around the surface that cannot be deformed into a point? (for example, in a sphere those are all the loops. But in the surface of a donut not all of them can)._
  
Those are some of the questions that you will have to answer to fully complete a level. The inhabitants of the surface itself will tell you their story, their worries and maybe some clue.
In particular, the goal of a level is to go walk along the surface and come back to the initial point. The game will tell you if the path you have walked is contractible or not. You will have to collect a few non contractible (independent) paths in order to go to the next level. This task is equivalent to find the generators of the fundamental group.
For extra rewards, the player should discover whether there is torsion (doing a loop several times makes it contractible) and what happens with the orientation.

There is **no needed background of Mathematics to play this game**. Moreover, you will learn a lot of Topology along the way!


This project is in a very early stage of development. So far, I have only completed a 'teaser', where you can walk around different surfaces (Torus, Moebius band, projective plane and Klein bottle) with very simple geometric structures and discover the counter intuitive consequences of orientation.


https://github.com/user-attachments/assets/a6c43905-9195-4240-ae0d-567f6f6be69f






## Project Structure

The project is planned as follows:
1. Engine that renders surfaces locally: TopEngine
2. Algorithm to compute homotopy deformations of loops
3. Game mechanics development
4. Level design
5. Minimalistic aesthetic design and music.

## TopEngine: the topological render

So far we are in part 1. We are developing an engine that processes locally topological information.
- A surface is given by a triangulation in a pure combinatorial way (note that the surface may not be embedded in $\mathbb{R}^3$)
- All the topological information (position of landmarks, orientations, etc) is stored in this simplicial complex.
- A riemannian metric is given in each triangle.

The engine knows exacly where is everything, and we walk around the surface. But the render only processes the information of a disk around us.
1. We find the correspondent disk on the surface
2. We load the topological information on that disk
3. We process the local geometry by solving a laplacian.


Math needed to design it:
- Topology of simplicial complexes and a bit of combinatorics
- Riemannian geometry and basic numerical methods to solve laplace equation
- Paralell transport and holonomy basics to correctly track the movement of the player in the surface
- Some algorithms in combinatorial homotopy theory, to compute deformations of loops.


If you have any suggestion or want to contribute somehow to the project, send an email to juanmartin.faj@gmail.com.
