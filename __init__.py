import bpy
import importlib
spam_loader = importlib.util.find_spec('cv2')
found = spam_loader is not None
if found:
    import cv2
import urllib
import time
import numpy
from bpy.props import EnumProperty
import webbrowser
from bpy_extras.io_utils import ImportHelper
import struct
import math

spam_loadertwo = importlib.util.find_spec('pyaudio')
foundaudio = spam_loadertwo is not None
if foundaudio:
    import pyaudio

bl_info = {
    "name": "jkirsons Open CV improved",
    "blender": (3, 0, 0),
    "category": "Object",
}

class VIEW3D_PT_TemplatePanel(bpy.types.Panel):
    bl_label = "CV2 face tracking"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode"
    bl_category = 'CV2FaceTracking'
    
    
    def draw(self, context):
        layout = self.layout
        if found and bpy.context.scene.cvtwo.landmark_model_path != "NO PATH SPECIFIED":
            layout = self.layout
            row = self.layout.row()
            if not hasattr(context.scene,"cv_target"):
                return 
            #if not context.scene.get(context.scene.cv_target):
            #    context.scene.cv_target == ""
            #if not context.scene.get(context.scene.cv_target, None).type != "ARMATURE":
            #    context.scene.cv_target == ""
            row = layout.row()
            row.label(text="Armature Target")
            object = row.prop_search(context.scene, 'cv_target', bpy.data, 'objects', icon='OBJECT_DATA', text='')
            try:
                bpy.data.objects[context.scene.cv_target].data.bones
            except Exception as e:
                row = layout.row()
                row.label(text="Need to select an object that is an Armature!")
                return
            pose_bone = bpy.data.objects[context.scene.cv_target].cvbonedata
            
            row = self.layout.row()
            if context.scene.cv_target == "":
                row.enabled = False
            operator_text = ""
            if bpy.context.scene.cvtwo.modalrunning == False:
                operator_text = "Start Tracking"
            else:
                operator_text = "Stop Tracking"
            op = row.operator("wm.opencv_operator", text=operator_text, icon="OUTLINER_OB_CAMERA")
            
            row = layout.row()
            row.prop(bpy.context.scene.cvtwo, "landmark_model_path")
            
            
            
            if foundaudio:
                row = self.layout.row()
                head_fk = row.prop(pose_bone, 'use_audio', text='Use audio for mouth movement')
                if pose_bone.use_audio == True:
                    row = self.layout.row()
                    row.prop(pose_bone, 'audio_channel')
            else:
                row = self.layout.row()
                row.label(text="Module pyaudio not installed. Can't use audio for mouth.")
                row = self.layout.row()
                head_fk = row.prop(pose_bone, 'use_audio', text='Use audio for mouth movement')
            row = self.layout.row()
            row.label(text="Head Bone")
            head_fk = row.prop_search(pose_bone, 'head_fk', bpy.data.objects[context.scene.cv_target].data, 'bones', icon='BONE_DATA', text='')
            
            row = self.layout.row()
            row.label(text="Mouth Control (x movement=width , y movement = height)")
            mouth_ctrl = row.prop_search(pose_bone, 'mouth_ctrl', bpy.data.objects[context.scene.cv_target].data, 'bones', icon='BONE_DATA', text='')
            
            row = self.layout.row()
            row.label(text="Eyebrow L")
            brow_ctrl_L = row.prop_search(pose_bone, 'brow_ctrl_L', bpy.data.objects[context.scene.cv_target].data, 'bones', icon='BONE_DATA', text='')
            
            row = self.layout.row()
            row.label(text="Eyebrow R")
            brow_ctrl_R = row.prop_search(pose_bone, 'brow_ctrl_R', bpy.data.objects[context.scene.cv_target].data, 'bones', icon='BONE_DATA', text='')
            
        else:
            row = layout.row()
            row.label(text="CV2 MODULE IS NOT INSTALLED ON THIS BLENDER VERSION!", icon="ERROR")
            row = layout.row()
            row.operator("wm.modellinkopener")
            row = layout.row()
            row.prop(bpy.context.scene.cvtwo, "landmark_model_path")

        
        

        #props = tool.operator_properties("wm.opencv_operator")
        #layout.prop(props, "stop", text="Stop Capture")
        #layout.prop(tool.op, "stop", text="Stop Capture")


class ArmatureData(bpy.types.PropertyGroup):
    # use an annotation
    head_fk : bpy.props.StringProperty()
    mouth_ctrl : bpy.props.StringProperty()
    brow_ctrl_L : bpy.props.StringProperty()
    brow_ctrl_R : bpy.props.StringProperty()
    use_audio : bpy.props.BoolProperty()
    audio_channel : bpy.props.IntProperty(default=0,description="Audio Channel For Microphone")
        
            
class CV2ModelLink(bpy.types.Operator):
    bl_label = "Open Trained Model Download Link"
    bl_idname = "wm.modellinkopener"
    
    def execute(self, context):
        webbrowser.open("https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml")
        

class FACIALTRACKING_OT_SelectObject(bpy.types.Operator):
    bl_label = "Start Face Tracking"
    bl_idname = "wm.facetrackmodal"
    
    preset_enum: bpy.props.EnumProperty(
        name = "option",
        description = "Choose A value to drive this object with.",
        items = [
            ("height", "Height", "Use height of mouth facial tracking here"),
            ("width", "Width", "Use width of mouth facial tracking here"),
            ("upperR", "Upper Right Eyelid Height", "Use Upper Right Eyelid Height facial tracking here"),
            ("lowerR", "Lower Right Eyelid Height","Use Lower Right Eyelid Height facial tracking here"),
            ("upperL", "Upper Left Eyelid Height", "Use Upper Left Eyelid Height facial tracking here"),
            ("lowerL", "Lower Left Eyelid Height","Use Lower Left Eyelid Height facial tracking here")   
        ]
    )
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self,"preset_enum")
    
    def execute(self, context):
        return {'FINISHED'}
     

# Download trained model (lbfmodel.yaml)
# https://github.com/kurnianggoro/GSOC2017/tree/master/data

# Install prerequisites:

# Linux: (may vary between distro's and installation methods)
# This is for manjaro with Blender installed from the package manager
# python3 -m ensurepip
# python3 -m pip install --upgrade pip --user
# python3 -m pip install opencv-contrib-python numpy --user

# MacOS
# open the Terminal
# cd /Applications/Blender.app/Contents/Resources/2.81/python/bin
# ./python3.7m -m ensurepip
# ./python3.7m -m pip install --upgrade pip --user
# ./python3.7m -m pip install opencv-contrib-python numpy --user

# Windows:
# Open Command Prompt as Administrator
# cd "C:\Program Files\Blender Foundation\Blender 2.81\2.81\python\bin"
# python -m pip install --upgrade pip
# python -m pip install opencv-contrib-python numpy

class OpenCVAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"
    
    # Set paths to trained models downloaded above
    face_detect_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    #landmark_model_path = "/home/username/Documents/Vincent/lbfmodel.yaml"  #Linux
    #landmark_model_path = "/Users/username/Downloads/lbfmodel.yaml"         #Mac
    
    
    # Load models
    fm = None
    cas = cv2.CascadeClassifier(face_detect_path)
        
    _timer = None
    _cap  = None
    
    
    p = None
    stream = None
    all = None
    aux = None
    
    
    
        
    bones = None
    Head = None
    Mouth = None
    BrowL = None
    BrowR = None
    
    # Webcam resolution:
    width = 640
    height = 480
    
    # 3D model points. 
    model_points = numpy.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ], dtype = numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )
                            
    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    # Keeps min and max values, then returns the value in a range 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0.0
        
    # The main "loop"
    def modal(self, context, event):
        
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or bpy.context.scene.cvtwo.modalrunning == False:
            bpy.context.scene.cvtwo.modalrunning = False
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera(context)
            _, image = self._cap.read()
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = cv2.equalizeHist(gray)
            
            # find faces
            faces = self.cas.detectMultiScale(image, 
                scaleFactor=1.05,  
                minNeighbors=3, 
                flags=cv2.CASCADE_SCALE_IMAGE, 
                minSize=(int(self.width/5), int(self.width/5)))
            
            #find biggest face, and only keep it
            if type(faces) is numpy.ndarray and faces.size > 0: 
                biggestFace = numpy.zeros(shape=(1,4))
                for face in faces:
                    if face[2] > biggestFace[0][2]:
                        #print(face)
                        biggestFace[0] = face
         
                # find the landmarks.
                _, landmarks = self.fm.fit(image, faces=biggestFace)
                for mark in landmarks:
                    shape = mark[0]
                    
                    #2D image points. If you change the image, you need to change vector
                    image_points = numpy.array([shape[30],     # Nose tip - 31
                                                shape[8],      # Chin - 9
                                                shape[36],     # Left eye left corner - 37
                                                shape[45],     # Right eye right corne - 46
                                                shape[48],     # Left Mouth corner - 49
                                                shape[54]      # Right mouth corner - 55
                                            ], dtype = numpy.float32)
                 
                    dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
                 
                    # determine head rotation
                    if hasattr(self, 'rotation_vector'):
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            rvec=self.rotation_vector, tvec=self.translation_vector, 
                            useExtrinsicGuess=True)
                    else:
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            useExtrinsicGuess=False)
                 
                    if not hasattr(self, 'first_angle'):
                        self.first_angle = numpy.copy(self.rotation_vector)
                    
                    try:
                        armature = bpy.context.scene.objects[context.scene.cv_target]
                        bones = armature.id_data.pose.bones
                    except:
                        return {'CANCELLED'}
                    
                    try:
                        self.Head = bones.get(armature.cvbonedata.head_fk)
                    except:
                        print("Head bone not found!")
                    
                    try:
                        self.Mouth = bones.get(armature.cvbonedata.mouth_ctrl)
                    except:
                        print("Mouth bone not found!")
                        
                    try:
                        self.BrowL = bones.get(armature.cvbonedata.brow_ctrl_L)
                    except:
                        print("Eyebrow L bone not found!")
                        
                    try:
                        self.BrowR = bones.get(armature.cvbonedata.brow_ctrl_R)
                    except:
                        print("Eyebrow R bone not found!")
                    
                    try:
                        self.Pitch = armature.cvbonedata.pitch
                    except:
                        print("no pitch! Using 0!")
                        self.Pitch = 0
                    
                    
                    
                    
                    if self.Head != None:
                        self.Head.rotation_euler[0] = self.smooth_value("h_x", 3, (self.rotation_vector[0] - self.first_angle[0])) / 1   # Up/Down
                        self.Head.rotation_euler[2] = self.smooth_value("h_y", 3, -(self.rotation_vector[1] - self.first_angle[1])) / 1.5  # Rotate
                        self.Head.rotation_euler[1] = self.smooth_value("h_z", 3, (self.rotation_vector[2] - self.first_angle[2])) / 1.3   # Left/Right
                        #Set keyframe so blender updates and shows position
                        self.Head.keyframe_insert(data_path="location", index=-1)
                    if self.Mouth != None:
                        if self.Audio and foundaudio:
                            data = self.audio_anaylisis()
                            self.Mouth.location[1] = self.smooth_value("m_h", 2, self.get_rms(data))
                        else:
                            self.Mouth.location[1] = self.smooth_value("m_h", 2, -self.get_range("mouth_height", numpy.linalg.norm(shape[62] - shape[66])) * 0.06 )
                            self.Mouth.location[0] = self.smooth_value("m_w", 2, (self.get_range("mouth_width", numpy.linalg.norm(shape[54] - shape[48])) - 0.5) * -0.04)
                        
                        #Set keyframe so blender updates and shows position
                        self.Mouth.keyframe_insert(data_path="location", index=-1)
                    if self.BrowL != None:
                        self.BrowL.location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", numpy.linalg.norm(shape[19] - shape[27])) -0.5) * 0.04)
                        #Set keyframe so blender updates and shows position
                        self.BrowL.keyframe_insert(data_path="location", index=2)
                    if self.BrowR != None:
                        self.BrowR.location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", numpy.linalg.norm(shape[24] - shape[27])) -0.5) * 0.04)
                        #Set keyframe so blender updates and shows position
                        self.BrowR.keyframe_insert(data_path="location", index=2)
                    
                    
                    
                        # draw face markers
                        #for (x, y) in shape:
                            #cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                
                # draw detected face
                #for (x,y,w,h) in faces:
                    #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
                

        return {'PASS_THROUGH'}
    
    def init_camera(self,context):
        if self._cap == None:
            
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)
            
            self.Head = None
            self.Mouth = None
            self.BrowL = None
            self.BrowR = None
            self.Audio = False
            
            
            if foundaudio:
                try:
                    armature = bpy.context.scene.objects[context.scene.cv_target]
                    self.Audio = armature.cvbonedata.use_audio
                except:
                    self.Audio = False
                
                if self.Audio:
                    #https://stackoverflow.com/questions/48653745/continuesly-streaming-audio-signal-real-time-infinitely-python
                    self.p = pyaudio.PyAudio()
                    self.stream = self.p.open(format = self.FORMAT,
                                    channels = self.CHANNELS,
                                    rate = self.RATE,
                                    input = True,
                                    frames_per_buffer = self.chunk,
                                    input_device_index = context.scene.objects[context.scene.cv_target].cvbonedata.audio_channel)

                    self.all = []
                    self.aux = []
                    self.stream.start_stream()
            
            
            
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        if bpy.context.scene.cvtwo.modalrunning == True:
            bpy.context.scene.cvtwo.modalrunning = False
            return {'FINISHED'}
        else:
            bpy.app.handlers.frame_change_pre.append(self.stop_playback)

            
            
                
            self.fm = cv2.face.createFacemarkLBF()
            self.fm.loadModel(context.scene.cvtwo.landmark_model_path)
            
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
            bpy.context.scene.cvtwo.modalrunning = True
            return {'RUNNING_MODAL'}
    
    
    if foundaudio:
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 4000
        ropen=True
        chunk = int(RATE/20) 
    
    
    #https://stackoverflow.com/questions/48653745/continuesly-streaming-audio-signal-real-time-infinitely-python
    def audio_anaylisis(self):
        data = self.stream.read(self.chunk)
        return data
    
    
    #https://stackoverflow.com/a/36413872
    def get_rms(self,block ):
        # RMS amplitude is defined as the square root of the 
        # mean over time of the square of the amplitude.
        # so we need to convert this string of bytes into 
        # a string of 16-bit samples...

        # we will get one short out for each 
        # two chars in the string.
        count = len(block)/2
        format = "%dh"%(count)
        shorts = struct.unpack( format, block )

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
            # sample is a signed short in +/- 32768. 
            # normalize it to 1.0
            n = sample * (1.0/32768.0)
            sum_squares += n*n

        return math.sqrt( sum_squares / count )
    
    
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
        if self.Audio and foundaudio:
            #https://stackoverflow.com/questions/48653745/continuesly-streaming-audio-signal-real-time-infinitely-python
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.all+=self.aux
            del  self.aux[:]

class CVTWOSettings(bpy.types.PropertyGroup):
    # use an annotation
    landmark_model_path : bpy.props.StringProperty(
        name="Full Trained Model Path",
        description="Must be a path to the model downloaded from the link in the addon.",
        default="NO PATH SPECIFIED",
        subtype='FILE_PATH'
        )
    modalrunning : bpy.props.BoolProperty(
        name="MODAL RUNNING INTERNAL DO NOT USE",
        default = False)


def register():
    bpy.utils.register_class(OpenCVAnimOperator)
    bpy.utils.register_class(FACIALTRACKING_OT_SelectObject) 
    bpy.utils.register_class(VIEW3D_PT_TemplatePanel)
    bpy.utils.register_class(CV2ModelLink)
    bpy.utils.register_class(CVTWOSettings)
    bpy.utils.register_class(ArmatureData)
    bpy.types.Scene.cv_target = bpy.props.StringProperty()
    bpy.types.Object.cvbonedata = bpy.props.PointerProperty(type=ArmatureData)
    bpy.types.Scene.cvtwo = bpy.props.PointerProperty(type=CVTWOSettings)
    #bpy.utils.register_tool(OBJECT_MT_OpenCVPanel, separator=True, group=True)

def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)
    bpy.utils.unregister_class(FACIALTRACKING_OT_SelectObject)
    bpy.utils.unregister_class(VIEW3D_PT_TemplatePanel)
    bpy.utils.unregister_class(CV2ModelLink)
    bpy.utils.unregister_class(CVTWOSettings)
    bpy.utils.unregister_class(ArmatureData)

if __name__ == "__main__":
    register()

    # test call
    #bpy.ops.wm.opencv_operator()
