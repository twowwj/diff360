from calendar import c
from functools import partial
import json
from unittest.mock import NonCallableMagicMock
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

import time
import cv2
import h5py
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import tqdm
import sys


class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.actor_id = self.smc.attrs['actor_id']
        self.performance_part = self.smc.attrs['performance_part']
        self.capture_date = self.smc.attrs['capture_date']
        self.actor_info = dict(
            age=self.smc.attrs['age'],
            color=self.smc.attrs['color'], 
            gender=self.smc.attrs['gender'],
            height=self.smc.attrs['height'], 
            weight=self.smc.attrs['weight'] 
            )
        self.Camera_info = dict(
            num_device=self.smc['Camera'].attrs['num_device'],
            num_frame=self.smc['Camera'].attrs['num_frame'],
            resolution=self.smc['Camera'].attrs['resolution'],
        )

    ###info 
    def get_actor_info(self):
        return self.actor_info
    
    def get_Camera_info(self):
        return self.Camera_info

    
    ### Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'00' ... '59'}
                Matrix_type in ['D', 'K', 'RT'] 
        """  
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__
        self.__calibration_dict__ = dict()
        for ci in self.smc['Calibration'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT'] :
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Calibration'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_id (int/str of a number):
                CameraID(str) in {'00' ... '60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT'] 
        """            
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc['Calibration'].keys(), f'Invalid Camera_id {Camera_id}'
        rs = dict()
        for k in ['D', 'K', 'RT'] :
            rs[k] = self.smc['Calibration'][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_img(self, Camera_id, Image_type, Frame_id=None, disable_tqdm=True):
        """Get image its Camera_id, Image_type and Frame_id
        
        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'00'...'59'}
            Image_type(str) in 
                    {'Camera': ['color','mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'-1 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
                'color': HWC(2048, 2448, 3) in bgr (uint8)
                'mask' : HW (2048, 2448) (uint8)
            multiple imgs :
                'color': NHWC(N, 2048, 2448, 3) in bgr (uint8)
                'mask' : NHW (N, 2048, 2448) (uint8)
        """ 
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f'Invalid Camera_id {Camera_id}'
        assert Image_type in self.smc["Camera"][Camera_id].keys(), f'Invalid Image_type {Image_type}'
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype {type(Frame_id)}'
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc["Camera"][Camera_id][Image_type].keys(), f'Invalid Frame_id {Frame_id}'
            if Image_type in ['color']:
                img_byte = self.smc["Camera"][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_color = np.max(img_color,2).astype(np.uint8)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc["Camera"][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_id, Image_type,fi))
            return np.stack(rs,axis=0)
    
    def get_audio(self):
        """
        Get audio data.
        Returns:
            a dictionary of audio data consists of:
                audio_np_array: np.ndarray
                sample_rate: int
        """
        if "s" not in self.performance_part.split('_')[0]:
            print(f"no audio data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Camera"]['00']['audio']
        return data
    
    def writemp3(self, f, sr, x, normalized=False):
        """numpy array to MP3"""
        channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(x * 2 ** 15)
        else:
            y = np.int16(x)
        song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
        song.export(f, format="mp3", bitrate="320k")

    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id,Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id
        PS: Not all the Camera_id/Frame_id have detected keypoints2d.

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in {18...32}
                    Not all the view have detection result, so the key will miss too when there are no lmk2d result
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            single lmk2d : (106, 2)
            multiple lmk2d : (N, 106, 2)
            if no data,return None
        """ 
        Camera_id = str(Camera_id)
        assert Camera_id in [f'%02d'%i for i in range(18,33)], f'Invalid Camera_id {Camera_id}'
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype: {type(Frame_id)}'
        if Camera_id not in self.smc['Keypoints2d'].keys():
            print(f"not lmk2d result in camera id {Camera_id}")
            return None
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            assert Frame_id >= 0 and Frame_id<self.smc['Keypoints2d'].attrs['num_frame'], f'Invalid frame_index {Frame_id}'
            Frame_id = str(Frame_id)
            if Frame_id not in self.smc['Keypoints2d'][Camera_id].keys() or \
                self.smc['Keypoints2d'][Camera_id][Frame_id] is None or \
                len(self.smc['Keypoints2d'][Camera_id][Frame_id]) == 0:
                print(f"not lmk2d result in Camera_id/Frame_id {Camera_id}/{Frame_id}")
                return None
            return self.smc['Keypoints2d'][Camera_id][Frame_id]
        else:
            if Frame_id is None:
                return self.smc['Keypoints2d'][Camera_id]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt2d = self.get_Keypoints2d(Camera_id,fi)
                if kpt2d is not None:
                    rs.append(kpt2d)
            return np.stack(rs,axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id
        PS: Not all the Frame_id have keypoints3d.

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
            if data do not exist: None
        """ 
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            assert Frame_id>=0 and Frame_id<self.smc['Keypoints3d'].attrs['num_frame'], \
                f'Invalid frame_index {Frame_id}'
            if str(Frame_id) not in self.smc['Keypoints3d'].keys() or \
                len(self.smc['Keypoints3d'][str(Frame_id)]) == 0:
                print(f"get_Keypoints3d: data of frame {Frame_id} do not exist.")
                return None
            return self.smc['Keypoints3d'][str(Frame_id)]
        else:
            if Frame_id is None:
                return self.smc['Keypoints3d']
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt3d = self.get_Keypoints3d(fi)
                if kpt3d is not None:
                    rs.append(kpt3d)
            return np.stack(rs,axis=0)

    ###FLAME
    def get_FLAME(self, Frame_id=None):
        """Get FLAME (world coordinate) computed by flame-fitting processing pipeline.
        FLAME is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                "global_pose"                   : double (3,)
                "neck_pose"                     : double (3,)
                "jaw_pose"                      : double (3,)
                "left_eye_pose"                 : double (3,)
                "right_eye_pose"                : double (3,)
                "trans"                         : double (3,)
                "shape"                         : double (100,)
                "exp"                           : double (50,)
                "verts"                         : double (5023,3)
                "albedos"                       : double (3,256,256)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no flame data in the performance part: {self.performance_part}")
            return None
        if "FLAME" not in self.smc.keys():
            print("not flame parameters, please check the performance part.")
            return None
        flame = self.smc['FLAME']
        if Frame_id is None:
            return flame
        elif isinstance(Frame_id, list):
            frame_list = [str(fi) for fi in Frame_id]
            rs = []
            for fi in  tqdm.tqdm(frame_list):
                rs.append(self.get_FLAME(fi))
            return np.stack(rs,axis=0)
        elif isinstance(Frame_id, (int,str)):
            Frame_id = int(Frame_id)
            assert Frame_id>=0 and Frame_id<self.smc['FLAME'].attrs['num_frame'], f'Invalid frame_index {Frame_id}'
            return flame[str(Frame_id)]      
        else:
            raise TypeError('frame_id should be int, list or None.')
    
    ###uv texture map
    def get_uv(self, Frame_id=None, disable_tqdm=True):
        """Get uv map (image form) computed by flame-fitting processing pipeline.
        uv texture is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            a single img: HWC in bgr (uint8)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no uv data in the performance part: {self.performance_part}")
            return None
        if "UV_texture" not in self.smc.keys():
            print("not uv texture, please check the performance part.")
            return None
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype {type(Frame_id)}'
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc['UV_texture'].keys(), f'Invalid Frame_id {Frame_id}'
            img_byte = self.smc['UV_texture'][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc['UV_texture'].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_uv(fi))
            return np.stack(rs,axis=0)
    
    ###scan mesh
    def get_scanmesh(self):
        """
        Get scan mesh data computed by Dense Mesh Reconstruction pipeline.
        Returns:
            dict:
                'vertex': np.ndarray of vertics point (n, 3)
                'vertex_indices': np.ndarray of vertex indices (m, 3)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no scan mesh data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Scan"]
        return data
    
    def write_ply(self, scan, outpath):
        from plyfile import PlyData, PlyElement
        vertex = np.empty(len(scan['vertex']), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(len(scan['vertex'])):
            vertex[i] = np.array([(scan['vertex'][i,0], scan['vertex'][i,1], scan['vertex'][i,2])], \
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        triangles = scan['vertex_indices']
        face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,)),
                           ('red', 'u1'), ('green', 'u1'),
                           ('blue', 'u1')])
        for i in range(len(triangles)):
            face[i] = np.array([
                ([triangles[i,0],triangles[i,1],triangles[i,2]], 255, 255, 255)
            ],
            dtype=[('vertex_indices', 'i4', (3,)),
                ('red', 'u1'), ('green', 'u1'),
                ('blue', 'u1')])
        PlyData([
                PlyElement.describe(vertex, 'vertex'),
                PlyElement.describe(face, 'face') 
                ], text=True).write(outpath)

    def get_scanmask(self, Camera_id=None):
        """Get image its Camera_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'00'...'59'}
        Returns:
            a single img : HW (2048, 2448) (uint8)
            multiple img: NHW (N, 2048, 2448)  (uint8)
        """ 
        if Camera_id is None:
            rs = []
            for i in range(60):
                rs.append(self.get_scanmask(f'{i:02d}'))
            return np.stack(rs, axis=0)
        assert isinstance(Camera_id, (str,int)), f'Invalid Camera_id type {Camera_id}'
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f'Invalid Camera_id {Camera_id}'
        img_byte = self.smc["ScanMask"][Camera_id][()]
        img_color = self.__read_color_from_bytes__(img_byte)
        img_color = np.max(img_color,2).astype(np.uint8)
        return img_color           

def generate_actor_data(actor_id=None, output_dir=None):
    """
    Generate data for an actor: 12 expressions × 60 viewpoints

    Args:
        actor_id (str): Actor ID (required)
        output_dir (str): Directory to save generated data
    """
    import os
    if actor_id is None:
        raise ValueError("actor_id must be provided")
    if output_dir is None:
        output_dir = f"./{actor_id}_generated_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use 12 expressions (e0 to e11)
    expressions = [f"e{i}" for i in range(12)]

    print(f"Generating data for actor {actor_id}")
    print(f"Expressions: {expressions}")
    print(f"Output directory: {output_dir}")

    # Process each expression
    for exp_idx, expression in enumerate(expressions):
        print(f"\n=== Processing Expression {expression} ({exp_idx+1}/12) ===")

        # Construct file path
        smc_file = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_{expression}_raw.smc"

        # Check if file exists
        if not os.path.exists(smc_file):
            print(f"Warning: File {smc_file} does not exist, skipping...")
            continue

        try:
            # Load SMC file
            st = time.time()
            print(f"Loading {smc_file}...")
            rd = SMCReader(smc_file)
            print(f"SMCReader loaded in {time.time() - st:.2f} seconds")

            # Get basic info
            actor_info = rd.get_actor_info()
            camera_info = rd.get_Camera_info()
            print(f"Actor info: {actor_info}")
            print(f"Camera info: {camera_info}")

            # Create expression directory
            exp_dir = os.path.join(output_dir, expression)
            os.makedirs(exp_dir, exist_ok=True)

            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Save basic info
            info_file = os.path.join(exp_dir, "info.json")
            with open(info_file, 'w') as f:
                json.dump({
                    'actor_info': convert_numpy_types(actor_info),
                    'camera_info': convert_numpy_types(camera_info),
                    'expression': expression,
                    'actor_id': actor_id
                }, f, indent=2)

            # Process all 60 camera viewpoints
            print(f"Processing 60 camera viewpoints...")

            # Get calibration data for all cameras
            print("Getting camera calibration data...")
            calibrations = rd.get_Calibration_all()

            # Save calibration data
            calib_file = os.path.join(exp_dir, "calibrations.json")
            # Convert numpy arrays to lists for JSON serialization
            calib_serializable = convert_numpy_types(calibrations)

            with open(calib_file, 'w') as f:
                json.dump(calib_serializable, f, indent=2)

            # Process images for each camera (60 viewpoints)
            images_dir = os.path.join(exp_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Get number of frames
            num_frames = camera_info['num_frame']
            print(f"Number of frames: {num_frames}")

            # For each camera viewpoint
            for cam_id in range(60):
                cam_str = f"{cam_id:02d}"
                print(f"Processing camera {cam_str}...")

                # Create camera directory
                cam_dir = os.path.join(images_dir, f"cam_{cam_str}")
                os.makedirs(cam_dir, exist_ok=True)

                # Get images for this camera (all frames)
                try:
                    # Get first frame as sample
                    sample_image = rd.get_img(cam_str, 'color', 0)

                    # Save sample image
                    sample_path = os.path.join(cam_dir, f"frame_000.jpg")
                    cv2.imwrite(sample_path, sample_image)

                    # Save image info
                    img_info = {
                        'camera_id': cam_str,
                        'image_shape': sample_image.shape,
                        'num_frames': num_frames,
                        'sample_frame_path': sample_path
                    }

                    info_path = os.path.join(cam_dir, "camera_info.json")
                    with open(info_path, 'w') as f:
                        json.dump(convert_numpy_types(img_info), f, indent=2)

                except Exception as e:
                    print(f"Error processing camera {cam_str}: {e}")
                    continue

            # Get FLAME parameters if available
            try:
                print("Getting FLAME parameters...")
                flame_data = rd.get_FLAME(0)  # Get first frame
                if flame_data is not None:
                    flame_dir = os.path.join(exp_dir, "flame")
                    os.makedirs(flame_dir, exist_ok=True)

                    # Save FLAME parameters (convert numpy to lists)
                    flame_serializable = convert_numpy_types(flame_data)

                    flame_file = os.path.join(flame_dir, "flame_frame_0.json")
                    with open(flame_file, 'w') as f:
                        json.dump(flame_serializable, f, indent=2)

                    print(f"FLAME parameters saved to {flame_file}")

            except Exception as e:
                print(f"Error getting FLAME parameters: {e}")

            # Get keypoints if available
            try:
                print("Getting 3D keypoints...")
                kpt3d = rd.get_Keypoints3d(0)  # Get first frame
                if kpt3d is not None:
                    kpt_dir = os.path.join(exp_dir, "keypoints")
                    os.makedirs(kpt_dir, exist_ok=True)

                    kpt_file = os.path.join(kpt_dir, "keypoints3d_frame_0.json")
                    with open(kpt_file, 'w') as f:
                        json.dump(convert_numpy_types(kpt3d), f, indent=2)

                    print(f"3D keypoints saved to {kpt_file}")

            except Exception as e:
                print(f"Error getting 3D keypoints: {e}")

            print(f"Expression {expression} processing completed!")

        except Exception as e:
            print(f"Error processing expression {expression}: {e}")
            continue

    print(f"\n=== Data generation completed! ===")
    print(f"Generated data saved to: {output_dir}")
    print(f"Structure: {output_dir}/[expression]/[data_type]/...")

def generate_actor_data_compact(actor_id=None, output_dir=None, include_speech=True, include_head=True):
    """
    Generate compact data for an actor: All data types × 60 viewpoints
    All images in one folder, all metadata in one JSON file

    Args:
        actor_id (str): Actor ID (required)
        output_dir (str): Directory to save generated data
        include_speech (bool): Whether to include speech data (s*)
        include_head (bool): Whether to include head data (h*)
    """
    import os
    if actor_id is None:
        raise ValueError("actor_id must be provided")
    if output_dir is None:
        output_dir = f"./{actor_id}_data_compact"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Collect all data types
    data_types = []

    # Add expressions (e0 to e11)
    expressions = [f"e{i}" for i in range(12)]
    data_types.extend(expressions)

    # Add speech data (s1 to s6) if requested
    if include_speech:
        speech_data = [f"s{i}_all" for i in range(1, 7)]
        data_types.extend(speech_data)

    # Add head data (h0) if requested
    if include_head:
        head_data = ["h0"]
        data_types.extend(head_data)

    print(f"Generating compact data for actor {actor_id}")
    print(f"Expressions: {expressions}")
    print(f"Output directory: {output_dir}")

    # Master data dictionary to store all metadata
    master_data = {
        "actor_id": actor_id,
        "actor_info": {},
        "expressions": {},
        "calibrations": {},
        "summary": {
            "total_expressions": len(expressions),
            "total_cameras": 60,
            "total_images": 0,
            "image_resolution": None
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    total_images = 0

    # Process each expression
    for exp_idx, expression in enumerate(expressions):
        print(f"\n=== Processing Expression {expression} ({exp_idx+1}/12) ===")

        # Construct file path
        smc_file = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_{expression}_raw.smc"

        # Check if file exists
        if not os.path.exists(smc_file):
            print(f"Warning: File {smc_file} does not exist, skipping...")
            continue

        try:
            # Load SMC file
            st = time.time()
            print(f"Loading {smc_file}...")
            rd = SMCReader(smc_file)
            print(f"SMCReader loaded in {time.time() - st:.2f} seconds")

            # Get basic info
            actor_info = rd.get_actor_info()
            camera_info = rd.get_Camera_info()
            print(f"Actor info: {actor_info}")
            print(f"Camera info: {camera_info}")

            # Store actor info (only once)
            if not master_data["actor_info"]:
                master_data["actor_info"] = convert_numpy_types(actor_info)
                master_data["summary"]["image_resolution"] = convert_numpy_types(camera_info["resolution"])

            # Store expression info
            master_data["expressions"][expression] = {
                "num_frames": convert_numpy_types(camera_info["num_frame"]),
                "cameras": {}
            }

            # Get calibration data for all cameras (only once)
            if not master_data["calibrations"]:
                print("Getting camera calibration data...")
                calibrations = rd.get_Calibration_all()
                master_data["calibrations"] = convert_numpy_types(calibrations)

            # Process images for each camera (60 viewpoints)
            print(f"Processing 60 camera viewpoints...")

            # Get number of frames
            num_frames = camera_info['num_frame']
            print(f"Number of frames: {num_frames}")

            # For each camera viewpoint
            for cam_id in range(60):
                cam_str = f"{cam_id:02d}"
                print(f"Processing camera {cam_str}...")

                try:
                    # Get first frame as sample
                    sample_image = rd.get_img(cam_str, 'color', 0)

                    # Save image with naming: {expression}_cam_{camera_id}_frame_000.jpg
                    image_filename = f"{expression}_cam_{cam_str}_frame_000.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, sample_image)

                    # Store camera info in master data
                    master_data["expressions"][expression]["cameras"][cam_str] = {
                        "image_filename": image_filename,
                        "image_shape": convert_numpy_types(sample_image.shape),
                        "image_path": f"images/{image_filename}"
                    }

                    total_images += 1

                except Exception as e:
                    print(f"Error processing camera {cam_str}: {e}")
                    continue

            # Try to get FLAME parameters if available
            try:
                print("Getting FLAME parameters...")
                flame_data = rd.get_FLAME(0)  # Get first frame
                if flame_data is not None:
                    master_data["expressions"][expression]["flame"] = convert_numpy_types(flame_data)
                    print(f"FLAME parameters stored for {expression}")

            except Exception as e:
                print(f"Error getting FLAME parameters: {e}")

            # Try to get keypoints if available
            try:
                print("Getting 3D keypoints...")
                kpt3d = rd.get_Keypoints3d(0)  # Get first frame
                if kpt3d is not None:
                    master_data["expressions"][expression]["keypoints3d"] = convert_numpy_types(kpt3d)
                    print(f"3D keypoints stored for {expression}")

            except Exception as e:
                print(f"Error getting 3D keypoints: {e}")

            print(f"Expression {expression} processing completed!")

        except Exception as e:
            print(f"Error processing expression {expression}: {e}")
            continue

    # Update summary
    master_data["summary"]["total_images"] = total_images

    # Save master data JSON
    master_json_path = os.path.join(output_dir, f"{actor_id}_data.json")
    print(f"\nSaving master data to {master_json_path}...")
    with open(master_json_path, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"\n=== Compact data generation completed! ===")
    print(f"Generated data saved to: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"  ├── {actor_id}_data.json  (All metadata)")
    print(f"  └── images/         (All {total_images} images)")

    return master_data

def generate_actor_h0_data(actor_id=None, output_dir=None):
    """
    Generate data specifically for actor h0 (head) data

    Args:
        actor_id (str): Actor ID (required)
        output_dir (str): Directory to save generated data
    """
    import os
    if actor_id is None:
        raise ValueError("actor_id must be provided")
    if output_dir is None:
        output_dir = f"./{actor_id}_h0_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    data_type = "h0"

    print(f"Generating h0 (head) data for actor {actor_id}")
    print(f"Output directory: {output_dir}")

    # Master data dictionary
    master_data = {
        "actor_id": actor_id,
        "data_type": data_type,
        "actor_info": {},
        "cameras": {},
        "calibrations": {},
        "summary": {
            "total_cameras": 60,
            "total_images": 0,
            "image_resolution": None
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Construct file path
    smc_file = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_{data_type}_raw.smc"

    # Check if file exists
    if not os.path.exists(smc_file):
        print(f"Error: File {smc_file} does not exist!")
        return None

    try:
        # Load SMC file
        st = time.time()
        print(f"Loading {smc_file}...")
        rd = SMCReader(smc_file)
        print(f"SMCReader loaded in {time.time() - st:.2f} seconds")

        # Get basic info
        actor_info = rd.get_actor_info()
        camera_info = rd.get_Camera_info()
        print(f"Actor info: {actor_info}")
        print(f"Camera info: {camera_info}")

        # Store actor info
        master_data["actor_info"] = convert_numpy_types(actor_info)
        master_data["summary"]["image_resolution"] = convert_numpy_types(camera_info["resolution"])

        # Get calibration data for all cameras
        print("Getting camera calibration data...")
        calibrations = rd.get_Calibration_all()
        master_data["calibrations"] = convert_numpy_types(calibrations)

        # Process images for each camera (60 viewpoints)
        print(f"Processing 60 camera viewpoints...")

        # Get number of frames
        num_frames = camera_info['num_frame']
        print(f"Number of frames: {num_frames}")

        total_images = 0

        # For each camera viewpoint
        for cam_id in range(60):
            cam_str = f"{cam_id:02d}"
            print(f"Processing camera {cam_str}...")

            try:
                # Get first frame as sample
                sample_image = rd.get_img(cam_str, 'color', 0)

                # Save image with naming: h0_cam_{camera_id}_frame_000.jpg
                image_filename = f"h0_cam_{cam_str}_frame_000.jpg"
                image_path = os.path.join(images_dir, image_filename)
                cv2.imwrite(image_path, sample_image)

                # Store camera info in master data
                master_data["cameras"][cam_str] = {
                    "image_filename": image_filename,
                    "image_shape": convert_numpy_types(sample_image.shape),
                    "image_path": f"images/{image_filename}",
                    "num_frames": convert_numpy_types(num_frames)
                }

                total_images += 1

            except Exception as e:
                print(f"Error processing camera {cam_str}: {e}")
                continue

        # Update summary
        master_data["summary"]["total_images"] = total_images

        # Try to get additional data if available
        try:
            print("Getting FLAME parameters...")
            flame_data = rd.get_FLAME(0)  # Get first frame
            if flame_data is not None:
                master_data["flame"] = convert_numpy_types(flame_data)
                print(f"FLAME parameters stored")

        except Exception as e:
            print(f"FLAME parameters not available: {e}")

        try:
            print("Getting 3D keypoints...")
            kpt3d = rd.get_Keypoints3d(0)  # Get first frame
            if kpt3d is not None:
                master_data["keypoints3d"] = convert_numpy_types(kpt3d)
                print(f"3D keypoints stored")

        except Exception as e:
            print(f"3D keypoints not available: {e}")

        # Save master data JSON
        master_json_path = os.path.join(output_dir, f"{actor_id}_h0_data.json")
        print(f"\nSaving master data to {master_json_path}...")
        with open(master_json_path, 'w') as f:
            json.dump(master_data, f, indent=2)

        print(f"\n=== H0 data generation completed! ===")
        print(f"Generated data saved to: {output_dir}")
        print(f"Structure:")
        print(f"  {output_dir}/")
        print(f"  ├── {actor_id}_h0_data.json  (All metadata)")
        print(f"  └── images/            (All {total_images} images)")

        return master_data

    except Exception as e:
        print(f"Error processing h0 data: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_actor_complete_data(actor_id=None, output_dir=None):
    """
    Generate complete data for actor: expressions + head data

    Args:
        actor_id (str): Actor ID (required)
        output_dir (str): Directory to save generated data
    """
    import os
    if actor_id is None:
        raise ValueError("actor_id must be provided")
    if output_dir is None:
        output_dir = f"./{actor_id}_complete_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Generating complete data for actor {actor_id}")
    print(f"Output directory: {output_dir}")

    # Master data dictionary
    master_data = {
        "actor_id": actor_id,
        "actor_info": {},
        "data_types": {
            "expressions": {},  # e0-e11
            "head": {}          # h0
        },
        "calibrations": {},
        "summary": {
            "total_expressions": 12,
            "total_head_data": 1,
            "total_cameras": 60,
            "total_images": 0,
            "image_resolution": None
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    total_images = 0

    # Process expressions (e0-e11)
    expressions = [f"e{i}" for i in range(12)]
    print(f"\n=== Processing {len(expressions)} expressions ===")

    for exp_idx, expression in enumerate(expressions):
        print(f"\nProcessing Expression {expression} ({exp_idx+1}/12)")

        # Construct file path
        smc_file = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_{expression}_raw.smc"

        if not os.path.exists(smc_file):
            print(f"Warning: File {smc_file} does not exist, skipping...")
            continue

        try:
            # Load SMC file
            rd = SMCReader(smc_file)

            # Get basic info (only once)
            if not master_data["actor_info"]:
                actor_info = rd.get_actor_info()
                camera_info = rd.get_Camera_info()
                master_data["actor_info"] = convert_numpy_types(actor_info)
                master_data["summary"]["image_resolution"] = convert_numpy_types(camera_info["resolution"])

                # Get calibration data (only once)
                calibrations = rd.get_Calibration_all()
                master_data["calibrations"] = convert_numpy_types(calibrations)

            # Store expression info
            master_data["data_types"]["expressions"][expression] = {
                "cameras": {}
            }

            # Process images for each camera
            for cam_id in range(60):
                cam_str = f"{cam_id:02d}"

                try:
                    # Get first frame as sample
                    sample_image = rd.get_img(cam_str, 'color', 0)

                    # Save image with naming: {expression}_cam_{camera_id}_frame_000.jpg
                    image_filename = f"{expression}_cam_{cam_str}_frame_000.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, sample_image)

                    # Store camera info
                    master_data["data_types"]["expressions"][expression]["cameras"][cam_str] = {
                        "image_filename": image_filename,
                        "image_shape": convert_numpy_types(sample_image.shape),
                        "image_path": f"images/{image_filename}"
                    }

                    total_images += 1

                except Exception as e:
                    print(f"Error processing camera {cam_str}: {e}")
                    continue

            print(f"Expression {expression} completed!")

        except Exception as e:
            print(f"Error processing expression {expression}: {e}")
            continue

    # Process head data (h0)
    print(f"\n=== Processing head data (h0) ===")

    smc_file = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_h0_raw.smc"

    if os.path.exists(smc_file):
        try:
            # Load SMC file
            rd = SMCReader(smc_file)

            # Store head data info
            master_data["data_types"]["head"]["h0"] = {
                "cameras": {}
            }

            # Process images for each camera
            for cam_id in range(60):
                cam_str = f"{cam_id:02d}"

                try:
                    # Get first frame as sample
                    sample_image = rd.get_img(cam_str, 'color', 0)

                    # Save image with naming: h0_cam_{camera_id}_frame_000.jpg
                    image_filename = f"h0_cam_{cam_str}_frame_000.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, sample_image)

                    # Store camera info
                    master_data["data_types"]["head"]["h0"]["cameras"][cam_str] = {
                        "image_filename": image_filename,
                        "image_shape": convert_numpy_types(sample_image.shape),
                        "image_path": f"images/{image_filename}"
                    }

                    total_images += 1

                except Exception as e:
                    print(f"Error processing camera {cam_str}: {e}")
                    continue

            print(f"Head data (h0) completed!")

        except Exception as e:
            print(f"Error processing head data: {e}")
    else:
        print(f"Warning: Head data file {smc_file} not found")

    # Update summary
    master_data["summary"]["total_images"] = total_images

    # Save master data JSON
    master_json_path = os.path.join(output_dir, f"{actor_id}_complete_data.json")
    print(f"\nSaving master data to {master_json_path}...")
    with open(master_json_path, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"\n=== Complete data generation finished! ===")
    print(f"Generated data saved to: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"  ├── {actor_id}_complete_data.json  (All metadata)")
    print(f"  └── images/                  (All {total_images} images)")
    print(f"")
    print(f"Data breakdown:")
    print(f"  - Expressions: {len(expressions)} (e0-e11)")
    print(f"  - Head data: 1 (h0)")
    print(f"  - Total cameras per data type: 60")
    print(f"  - Total images: {total_images}")

    return master_data

def extract_actor_all_data(actor_id=None, output_dir=None):
    """
    Extract ALL actor data: expressions (e0-e11) + head (h0) + speech (s1-s6)

    Args:
        actor_id (str): Actor ID (required)
        output_dir (str): Directory to save extracted data
    """
    import os
    import glob
    if actor_id is None:
        raise ValueError("actor_id must be provided")
    if output_dir is None:
        output_dir = f"./{actor_id}_extracted_all"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"=" * 80)
    print(f"🚀 EXTRACTING ALL {actor_id} DATA")
    print(f"=" * 80)
    print(f"Actor ID: {actor_id}")
    print(f"Output directory: {output_dir}")

    # Find all .smc files in actor directory
    smc_pattern = f"/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_id}/{actor_id}_*.smc"
    smc_files = glob.glob(smc_pattern)
    smc_files.sort()

    print(f"\n📁 Found {len(smc_files)} SMC files:")
    for smc_file in smc_files:
        filename = os.path.basename(smc_file)
        print(f"   {filename}")

    # Master data dictionary
    master_data = {
        "actor_id": actor_id,
        "extraction_info": {
            "total_smc_files": len(smc_files),
            "data_types": {
                "expressions": [],
                "head": [],
                "speech": []
            }
        },
        "actor_info": {},
        "data_details": {},
        "calibrations": {},
        "summary": {
            "total_cameras": 60,
            "total_images": 0,
            "total_data_types": 0,
            "image_resolution": None
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    total_images = 0
    processed_files = 0

    # Process each SMC file
    for smc_file in smc_files:
        filename = os.path.basename(smc_file)

        # Extract data type from filename: {actor_id}_XX_raw.smc
        try:
            data_type = filename.split('_')[1]  # e0, e1, h0, s1_all, etc.
            if data_type.endswith('_raw'):
                data_type = data_type[:-4]  # Remove '_raw'
        except:
            print(f"⚠️  Cannot parse data type from {filename}, skipping...")
            continue

        print(f"\n📊 Processing: {filename} (Data type: {data_type})")

        try:
            # Load SMC file
            st = time.time()
            rd = SMCReader(smc_file)
            load_time = time.time() - st
            print(f"   ✅ Loaded in {load_time:.2f} seconds")

            # Get basic info (only once)
            if not master_data["actor_info"]:
                actor_info = rd.get_actor_info()
                camera_info = rd.get_Camera_info()
                master_data["actor_info"] = convert_numpy_types(actor_info)
                master_data["summary"]["image_resolution"] = convert_numpy_types(camera_info["resolution"])

                # Get calibration data (only once)
                calibrations = rd.get_Calibration_all()
                master_data["calibrations"] = convert_numpy_types(calibrations)
                print(f"   📷 Camera info: {camera_info['num_device']} cameras, {camera_info['num_frame']} frames")
            else:
                camera_info = rd.get_Camera_info()
                print(f"   📷 Camera info: {camera_info['num_device']} cameras, {camera_info['num_frame']} frames")

            # Categorize data type
            if data_type.startswith('e'):
                category = "expressions"
                master_data["extraction_info"]["data_types"]["expressions"].append(data_type)
            elif data_type.startswith('h'):
                category = "head"
                master_data["extraction_info"]["data_types"]["head"].append(data_type)
            elif data_type.startswith('s'):
                category = "speech"
                master_data["extraction_info"]["data_types"]["speech"].append(data_type)
            else:
                category = "unknown"
                print(f"   ⚠️  Unknown data type category: {data_type}")

            # Initialize data details
            master_data["data_details"][data_type] = {
                "category": category,
                "filename": filename,
                "num_cameras": camera_info['num_device'],
                "num_frames": camera_info['num_frame'],
                "cameras": {}
            }

            # Extract images for each camera (first frame only for efficiency)
            num_cameras = camera_info['num_device']
            num_frames = camera_info['num_frame']

            data_type_images = 0

            for cam_id in range(num_cameras):
                cam_str = f"{cam_id:02d}"

                try:
                    # Get first frame
                    sample_image = rd.get_img(cam_str, 'color', 0)

                    # Save image with naming: {data_type}_cam_{camera_id}_frame_000.jpg
                    image_filename = f"{data_type}_cam_{cam_str}_frame_000.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, sample_image)

                    # Store camera info
                    master_data["data_details"][data_type]["cameras"][cam_str] = {
                        "image_filename": image_filename,
                        "image_shape": convert_numpy_types(sample_image.shape),
                        "image_path": f"images/{image_filename}",
                        "num_frames": convert_numpy_types(num_frames)
                    }

                    data_type_images += 1
                    total_images += 1

                except Exception as e:
                    print(f"   ❌ Error processing camera {cam_str}: {e}")
                    continue

            print(f"   ✅ Extracted {data_type_images} images from {data_type}")
            processed_files += 1

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            continue

    # Update summary
    master_data["summary"]["total_images"] = total_images
    master_data["summary"]["total_data_types"] = processed_files

    # Save master data JSON
    master_json_path = os.path.join(output_dir, f"{actor_id}_extracted_all.json")
    print(f"\n💾 Saving master data to {master_json_path}...")
    with open(master_json_path, 'w') as f:
        json.dump(master_data, f, indent=2)

    # Generate summary report
    summary_path = os.path.join(output_dir, "extraction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"{actor_id} Complete Data Extraction Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Actor ID: {actor_id}\n")
        f.write(f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total SMC files processed: {processed_files}/{len(smc_files)}\n")
        f.write(f"Total images extracted: {total_images}\n\n")

        f.write("Data Type Breakdown:\n")
        f.write("-" * 30 + "\n")

        expressions = master_data["extraction_info"]["data_types"]["expressions"]
        head_data = master_data["extraction_info"]["data_types"]["head"]
        speech_data = master_data["extraction_info"]["data_types"]["speech"]

        f.write(f"Expressions ({len(expressions)}): {', '.join(expressions)}\n")
        f.write(f"Head data ({len(head_data)}): {', '.join(head_data)}\n")
        f.write(f"Speech data ({len(speech_data)}): {', '.join(speech_data)}\n\n")

        f.write("Images per data type:\n")
        f.write("-" * 30 + "\n")
        for data_type, details in master_data["data_details"].items():
            num_images = len(details["cameras"])
            f.write(f"{data_type}: {num_images} images\n")

    print(f"\n" + "=" * 80)
    print(f"🎉 EXTRACTION COMPLETED!")
    print(f"=" * 80)
    print(f"✅ Processed: {processed_files}/{len(smc_files)} SMC files")
    print(f"🖼️  Extracted: {total_images} images")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Summary: {summary_path}")

    # Show breakdown
    expressions = master_data["extraction_info"]["data_types"]["expressions"]
    head_data = master_data["extraction_info"]["data_types"]["head"]
    speech_data = master_data["extraction_info"]["data_types"]["speech"]

    print(f"\n📊 Data Type Breakdown:")
    print(f"   🎭 Expressions ({len(expressions)}): {', '.join(expressions)}")
    print(f"   👤 Head data ({len(head_data)}): {', '.join(head_data)}")
    print(f"   🎵 Speech data ({len(speech_data)}): {', '.join(speech_data)}")

    return master_data

### test func
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate_actor":
            # Generate actor data (original version)
            if len(sys.argv) < 3:
                print("Usage: python renderme_360_reader.py generate_actor <actor_id> [output_dir]")
            else:
                actor_id = sys.argv[2]
                output_dir = f"./{actor_id}_generated_data" if len(sys.argv) < 4 else sys.argv[3]
                generate_actor_data(actor_id=actor_id, output_dir=output_dir)
        elif sys.argv[1] == "generate_actor_compact":
            # Generate actor data (compact version)
            if len(sys.argv) < 3:
                print("Usage: python renderme_360_reader.py generate_actor_compact <actor_id> [output_dir]")
            else:
                actor_id = sys.argv[2]
                output_dir = f"./{actor_id}_data_compact" if len(sys.argv) < 4 else sys.argv[3]
                generate_actor_data_compact(actor_id=actor_id, output_dir=output_dir)
        elif sys.argv[1] == "generate_actor_h0":
            # Generate actor h0 (head) data
            if len(sys.argv) < 3:
                print("Usage: python renderme_360_reader.py generate_actor_h0 <actor_id> [output_dir]")
            else:
                actor_id = sys.argv[2]
                output_dir = f"./{actor_id}_h0_data" if len(sys.argv) < 4 else sys.argv[3]
                generate_actor_h0_data(actor_id=actor_id, output_dir=output_dir)
        elif sys.argv[1] == "generate_actor_complete":
            # Generate actor complete data (expressions + head)
            if len(sys.argv) < 3:
                print("Usage: python renderme_360_reader.py generate_actor_complete <actor_id> [output_dir]")
            else:
                actor_id = sys.argv[2]
                output_dir = f"./{actor_id}_complete_data" if len(sys.argv) < 4 else sys.argv[3]
                generate_actor_complete_data(actor_id=actor_id, output_dir=output_dir)
        elif sys.argv[1] == "extract_actor_all":
            # Extract all actor data (expressions + head + speech)
            if len(sys.argv) < 3:
                print("Usage: python renderme_360_reader.py extract_actor_all <actor_id> [output_dir]")
            else:
                actor_id = sys.argv[2]
                output_dir = f"./{actor_id}_extracted_all" if len(sys.argv) < 4 else sys.argv[3]
                extract_actor_all_data(actor_id=actor_id, output_dir=output_dir)
        else:
            # Original test functionality
            actor_part = sys.argv[1]
            st = time.time()
            print("reading smc: {}".format(actor_part))
            rd = SMCReader(f'/home/weijie.wang/project/DiffPortrait360/Renderme360/{actor_part}.smc')
            print("SMCReader done, in %f sec\n" % ( time.time() - st ), flush=True)

            # basic info
            print(rd.get_actor_info())
            print(rd.get_Camera_info())

            # img
            Camera_id = "25"
            Frame_id = 0

            image = rd.get_img(Camera_id, 'color', Frame_id)          # Load image for the specified camera and frame
            print(f"image.shape: {image.shape}") #(2048, 2448, 3)
            images = rd.get_img(Camera_id,'color',disable_tqdm=False)
            print(f'color {images.shape}, {images.dtype}')

            # camera
            cameras = rd.get_Calibration_all()
            print(f"all_calib 30 RT: {cameras['30']['RT']}")
            camera = rd.get_Calibration(15)
            print(' split_calib ',camera)

            # audio
            if '_s' in actor_part:
                audio = rd.get_audio()
                print('audio', audio['audio'].shape, 'sample_rate', np.array(audio['sample_rate']))
                sr = int(np.array(audio['sample_rate'])); arr = np.array(audio['audio'])
                rd.writemp3(f='./test.mp3', sr=sr, x=arr, normalized=True)

            # landmark
            lmk2d = rd.get_Keypoints2d('25',4)
            print('kepoint2d',lmk2d.shape)
            lmk2ds = rd.get_Keypoints2d('26', [1,2,3,4,5])
            print(f"lmk2ds.shape: {lmk2ds.shape}")
            lmk3d = rd.get_Keypoints3d(4)
            print(f'kepoint3d shape: {lmk3d.shape}')
            lmk3d = rd.get_Keypoints3d([1,2,3,4,5])
            print(f'kepoint3d shape: {lmk3d.shape}')

            # flame
            if '_e' in actor_part:
                flame = rd.get_FLAME(56)
                print(f"keys: {flame.keys()}")
                print(f"global_pose: {flame['global_pose'].shape}")
                print(f"neck_pose: {flame['neck_pose'].shape}")
                print(f"jaw_pose: {flame['jaw_pose'].shape}")
                print(f"left_eye_pose: {flame['left_eye_pose'].shape}")
                print(f"right_eye_pose: {flame['right_eye_pose'].shape}")
                print(f"trans: {flame['trans'].shape}")
                print(f"shape: {flame['shape'].shape}")
                print(f"exp: {flame['exp'].shape}")
                print(f"verts: {flame['verts'].shape}")
                print(f"albedos: {flame['albedos'].shape}")
                flame = rd.get_FLAME()
                print(f"keys: {flame.keys()}")

            # uv texture
            if '_e' in actor_part:
                uv = rd.get_uv(Frame_id)
                print(f"uv shape: {uv.shape}")
                uv = rd.get_uv()
                print(f"uv shape: {uv.shape}")

            # scan mesh
            if '_e' in actor_part:
                scan = rd.get_scanmesh()
                print(f"keys: {scan.keys()}")
                print(f"vertex: {scan['vertex'].shape}")
                print(f"vertex_indices: {scan['vertex_indices'].shape}")
                rd.write_ply(scan, './test_scan.ply')

            # scan mask
            if '_e' in actor_part:
                scanmask = rd.get_scanmask('03')
                print(f"scanmask.shape: {scanmask.shape}")
                scanmask = rd.get_scanmask()
                print(f"scanmask.shape all: {scanmask.shape}")
    else:
        print("Usage:")
        print("  python renderme_360_reader.py generate_actor <actor_id> [output_dir]         # Generate actor expressions (original)")
        print("  python renderme_360_reader.py generate_actor_compact <actor_id> [output_dir] # Generate actor expressions (compact)")
        print("  python renderme_360_reader.py generate_actor_h0 <actor_id> [output_dir]      # Generate actor h0 (head) data")
        print("  python renderme_360_reader.py generate_actor_complete <actor_id> [output_dir]# Generate actor complete (expressions + head)")
        print("  python renderme_360_reader.py extract_actor_all <actor_id> [output_dir]      # Extract ALL actor data (expressions + head + speech)")
        print("  python renderme_360_reader.py <actor_part>                                    # Original test functionality")
