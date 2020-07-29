import taichi as ti
import numpy as np
ti.init(debug=True)

max_num_particles = 256

dt = 1e-3

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom_y = 0.05

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
R_x=ti.Vector(2,dt=ti.f32,shape=max_num_particles)
subStep_x=ti.Vector(2,dt=ti.f32,shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
R_v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
subStep_v=ti.Vector(2,dt=ti.f32,shape=max_num_particles)
A=ti.Matrix(2,2,dt=ti.f32,shape=(max_num_particles,max_num_particles))
b=ti.Vector(2, dt=ti.f32, shape=max_num_particles)
new_v=ti.Vector(2, dt=ti.f32, shape=max_num_particles)
solve_v=ti.Vector(2, dt=ti.f32, shape=max_num_particles)
total_force=ti.Vector(2, dt=ti.f32, shape=max_num_particles)
# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15

gravity = [0, -9.8]
#jacobian
K=ti.Matrix(2,2,dt=ti.f32,shape=(max_num_particles,max_num_particles))
M=ti.Matrix(2,2,dt=ti.f32,shape=(max_num_particles,max_num_particles))
@ti.func
# 此次采样的权重和下次采样的步长
def Runge_KuttaIter(weight,step):
    n = num_particles[None]
    # calculate first and sample second
    for i in range(n):
        # Gravity
        total_force[i] = ti.Vector(gravity) * particle_mass
        # spring
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = R_x[i] - R_x[j]
                total_force[i] += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
    for i in range(n):
        a = total_force[i] / particle_mass
        # average
        subStep_x[i] += weight / 6.0 * dt * R_v[i]
        subStep_v[i] += weight / 6.0 * dt * a
        # next subStep sample
        R_x[i] = x[i] + step * dt * R_v[i]
        R_v[i] = v[i] + step * dt * a



@ti.kernel
def Runge_Kutta():

    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None])  # damping
        # 最开始采样点为当前点
        R_x[i] = x[i]
        R_v[i] = v[i]
        subStep_x[i] = x[i]
        subStep_v[i] = v[i]

    Runge_KuttaIter(1,0.5)
    Runge_KuttaIter(2,0.5)
    Runge_KuttaIter(2,1)
    Runge_KuttaIter(1,0)

    # calculate first and sample second
    for i in range(n):
        x[i]=subStep_x[i]
        v[i]=subStep_v[i]
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
@ti.kernel
def Verlet():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        R_x[i]=x[i]
        v[i] *= ti.exp(-dt * damping[None])  # damping
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        R_x[i] += v[i] * dt+0.5*dt*dt*total_force / particle_mass
        if R_x[i].y < bottom_y:
            R_x[i].y = bottom_y
        v[i] = (R_x[i]-x[i])/dt
    for i in range(n):
        x[i]=R_x[i]


@ti.kernel
def substep():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        v[i] += dt * total_force / particle_mass
        
    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt


        
@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1

@ti.kernel
def implicit():
    n = num_particles[None]
    for i,j in ti.ndrange(n,n):
        M[i,j].fill(0)
        K[i,j].fill(0)
        A[i,j].fill(0)
        b[i].fill(0)
    for i in range(n):
        M[i,i][0,0]=particle_mass
        M[i,i][1,1]=particle_mass
        v[i] *= ti.exp(-dt * damping[None]) # damping
    for i,j in ti.ndrange(n,n):
            if rest_length[i, j] != 0:
                x_ij = x[j] - x[i]
                I=ti.Matrix([[1, 0], [0, 1]])
                l=x_ij.norm()
                t=spring_stiffness[None]*(-I+rest_length[i,j]/l*(I-(x_ij@x_ij.transpose()/(l*l))))
                K[i,i]+=t
                K[i,j]+=-t
    for i,j in ti.ndrange(n,n):
        A[i,j]=(M[i,j]-dt*dt*K[i,j])
    for i in range(n):
        b[i]=M[i,i]@v[i]+dt*ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                b[i] += -dt*spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
    for i in range(n):
        solve_v[i]=v[i]

@ti.kernel
def Solve():
    n = num_particles[None]
    for i in range(n):
        r=b[i]
        for j in range(n):
            if i!=j:
                r-=A[i,j]@solve_v[j]
        r_i=r[0]-A[i,i][0,1]*solve_v[i][1]
        new_v[i][0] = r_i / max(A[i, i][0, 0],1e-4)
        r_i=r[1]-A[i,i][1,0]*solve_v[i][0]
        new_v[i][1]=r_i/max(A[i,i][1,1],1e-4)
    for i in range(n):
        solve_v[i]=new_v[i]



@ti.kernel
def residual() -> ti.f32:
    res = 0.0
    n = num_particles[None]
    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] @ solve_v[j]
        res += r.dot( r)
    return res
@ti.kernel
def ImplicitUpdate():
    n = num_particles[None]
    for i in range(n):
        v[i]=solve_v[i]
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
     # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt
    
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

spring_stiffness[None] = 10000
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1
                
    if not paused[None]:
        for step in range(10):
            implicit()
            for i in range(20):
                Solve()
            ImplicitUpdate()
            # RungeKutta
            #Runge_Kutta()
            # Verlet
            # Verlet()
            # explicit
            #substep()

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()

