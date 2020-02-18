from itertools import cycle
from PIL import ImageTk, Image
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk     ## Python 2.x
else:
    import tkinter as tk     ## Python 3.x
from tkinter.filedialog import asksaveasfile 
from tkinter import messagebox,Menu
from anndata import AnnData
from stlearn._compat import Literal
from typing import Optional

import matplotlib.pyplot as plt

class ChangeImage():
    def __init__(self, root, adata, method):
        self.adata = adata
        self.method = method
        self.pil_imgs = []
        self.photos=[]
        self.routes = []
        self.load_images(root)
        self.image_num=0
        self.canvas = tk.Canvas(root, width=self.pil_imgs[0].size[0], 
                                height=self.pil_imgs[0].size[1],bg="#fff",bd=0)
        self.canvas.pack(fill=tk.BOTH,expand=True,side=tk.TOP)  
        self.canvas.bind('<Configure>', self._resize_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, 
                    anchor=tk.NW, image=self.photos[self.image_num])

        ## click on button to change image
        self.bt_plot = tk.Button(root, text="Plot!", 
                                command=lambda:[plot(self.adata,
                                S=int(self.routes[self.image_num][1])),root.destroy()],
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_plot.pack(side=tk.RIGHT, padx=5, expand=True)

        menu = Menu(root,bg="#fff",bd=0)
        root.config(menu=menu,width = 20,bg="#fff",bd=0)
        filemenu = Menu(menu)
        menu.add_cascade(label="  File  ", menu=filemenu)
        filemenu.add_command(label="  Save  ", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="  Exit  ", command=root.destroy)


        self.bt_next = tk.Button(root, text="Next", command=self.next_image,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_next.pack(side=tk.RIGHT,expand=True)
        self.bt_next.config( height = 1, width = 5 )
        self.bt_prev = tk.Button(root, text="Prev", command=self.previous_image,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_prev.pack(side=tk.LEFT,expand=True)
        self.bt_prev.config( height = 1, width = 5 )
        self.bt_select = tk.Button(root, text="Select", command=self.select_images,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_select.pack(fill=tk.X, padx=5, expand=True)
        self.bt_select.config( height = 1, width = 5 )
        self.frame1 = tk.Frame(root,bg="#fff")
        self.frame1.pack(fill=tk.X)
        self.lbl1 = tk.Label(self.frame1, text=("Root node: "), width=15,bg="#fff")
        self.lbl1.pack(side=tk.LEFT, padx=5, pady=5)
        self.v = tk.StringVar()
        self.v.set(self.adata.uns["pseudotimespace"]["routes"][self.routes[self.image_num]])

        self.lbl2 = tk.Label(self.frame1, text=(self.routes[self.image_num]), width=15,
                            bg="#fff",bd=0)
        self.lbl2.pack(side=tk.RIGHT, padx=5, pady=5)

        self.entry1 = tk.Entry(self.frame1,textvariable=self.v,bg="#dedede",bd=0)
        self.entry1.bind('<Return>', self.select_images)
        self.entry1.pack(fill=tk.X, padx=5, expand=True)
        #self.bt_exit = tk.Button(root, text="Exit", bg="orange", command=root.destroy)
        #self.bt_exit.pack(side=tk.LEFT)
    def next_image(self):
        self.image_num += 1
        if self.image_num >= len(self.photos):
            self.image_num=0
        ## pipe the next image to be displayed to the button
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
        self.v.set(self.adata.uns["pseudotimespace"]["routes"][self.routes[self.image_num]])
        
        self.lbl2['text'] = self.routes[self.image_num]
    def previous_image(self):
        self.image_num = self.image_num - 1
        if self.image_num < 0:
            self.image_num=len(self.photos)-1
        ## pipe the next image to be displayed to the button
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
        self.v.set(self.adata.uns["pseudotimespace"]["routes"][self.routes[self.image_num]])
        
        self.lbl2['text'] = self.routes[self.image_num]
        
    def load_images(self,root):
        """ copy data images to a list that is an instance variable
            all images are hard-coded here and so use data=
            instead of file=
        """
        ## put the images in an instance object (self.) so they aren't destroyed
        ## when the function exits and can be used anywhere in the class
        for item in self.adata.uns['plots']['trajectories'].items():
            self.pil_imgs.append(Image.fromarray(item[1]))
            self.photos.append(ImageTk.PhotoImage(image=Image.fromarray(item[1]),master=root))
            self.routes.append(item[0])
    def select_images(self,*args):
        if ((len(self.v.get()) == 0) or (int(self.v.get()) not in self.adata.uns['pseudotimespace']['flat_tree'].nodes)):
            messagebox.showerror("Error","Please choose available root! (" +
                     str(self.adata.uns['pseudotimespace']['flat_tree'].nodes) + ")")
            return None


        self.image_num = int(self.v.get())-1
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
    
    def save(self): 
        f = asksaveasfile(filetypes=(("Portable Network Graphics (*.png)", "*.png"),
                                        ("All Files (*.*)", "*.*")),
                             mode='wb',
                             defaultextension='.png')
        if f is None:
            return

        filename = f.name
        extension = filename.rsplit('.', 1)[-1]
        img = Image.fromarray(self.adata.uns['plots']["trajectories"][self.routes[self.image_num]])
        img.save(f, extension)
        f.close()
        
    def _resize_image(self,event):

        self.new_width = event.width
        self.new_height = event.height
        

        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
        

def pseudotimespace_plot(
    adata: AnnData,
    method: str = "trajectories",
    root: int = None,
    S: int = None,
    ):
    
    if ((S is None) and (root is None)):
    
        root=tk.Tk()
        root.title('stLearn - Trajectories Inferences - Pseudo-time-space')
        CI=ChangeImage(root,adata,method)
        root.mainloop()

    elif ((S is not None) and (root is not None)):
        raise ValueError(
            'Only provide one of the optional parameters "S" or "root" ')
    else:
        pts_plot(adata,root=root,S=S)



def pts_plot(
    adata: AnnData,
    method: str = "trajectories",
    dpi: int = 192,):
    
    if method not in ["trajectories"]:
        raise ValueError("Please provide one of these methods : fa, ica or ldvae")

    if method not in str(adata.obsm):
        raise ValueError("Please run trajectories pipeline")
        

    root=tk.Tk()
    root.title('stLearn - Pseudo-time-space - Trajectories')
    CI=ChangeImage(root,adata,method)
    root.mainloop()



def plot(
    adata: AnnData,
    method: str = "trajectories",
    root: int = None,
    S: int = None,
    dpi: int = 192,
    copy: bool = False,
) -> Optional[AnnData]:
    

    if ((S is not None) and (root is not None)):
        raise ValueError(
            'Only provide one of the optional parameters "S" or "root" ')

    plt.rcParams['figure.dpi'] = dpi

    if S is not None:
        
        plt.imshow(adata.uns["plots"]["trajectories"]["S" + str(S) + "_pseudotime"])
        plt.axis('off')
        plt.show()
    else:
        dictionary = adata.uns["pseudotimespace"]["routes"]

        route = find_routes(root,dictionary)

        plt.axis('off')
        plt.imshow(adata.uns["plots"]["trajectories"][route])
        
        plt.show()





def find_routes(pattern,dictionary):
    
    for route, root in dictionary.items():
        if root == pattern:
            return route