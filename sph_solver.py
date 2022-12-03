print("-------------------------------")

print("Importing the SPH solver module")
import importlib
import particle_system
importlib.reload(particle_system)

class Solver():
    print("Importing Solver class")

    solver = None
    ps = None
    config = None
    def init(self):
        import os
        import argparse
        import taichi as ti
        import numpy as np
        from config_builder import SimConfig
        from particle_system import ParticleSystem
        print("Initilizing the SPH solver")

        ti.init(arch=ti.gpu, device_memory_fraction=0.5)

        parser = argparse.ArgumentParser(description='SPH Taichi')
        parser.add_argument('--scene_file',
                            default='E:/Dev/SPH_Taichi/data/scenes/sphere_dance.json',
                            help='scene file')
        args = parser.parse_args()
        scene_path = args.scene_file
        self.config = SimConfig(scene_file_path=scene_path)
        scene_name = scene_path.split("/")[-1].split(".")[0]

        substeps = self.config.get_cfg("numberOfStepsPerRenderUpdate")
        output_frames = self.config.get_cfg("exportFrame")
        output_interval = int(0.016 / self.config.get_cfg("timeStepSize"))
        output_ply = self.config.get_cfg("exportObj")
        series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
        if output_frames:
            os.makedirs(f"{scene_name}_output_img", exist_ok=True)
        if output_ply:
            os.makedirs(f"{scene_name}_output", exist_ok=True)

        self.ps = ParticleSystem(self.config, GGUI=True)
        self.solver = self.ps.build_solver()
        self.solver.initialize()

        self.ps.input_data()
        # self.solver = solver
        # self.ps = ps
        # self.config = config
        # return solver, ps, config

    def update(self,frame):
        print("Calling the solve()")
        if self.solver!=None:

            print("current frame: ", frame)
            self.ps.update_data()
            self.solver.step(frame)

            if self.config.get_cfg("exportPly") == True:
                obj_id = 0
                obj_data = self.ps.dump(obj_id=obj_id)
                np_pos = obj_data["position"]
                print(np_pos[1])
        else:
            print("Solver is undefined! Please call init() first")
    

# def test():
#     solver1 = Sph()
#     solver1.solve(1)

# if __name__ == "__main__":
#     test()