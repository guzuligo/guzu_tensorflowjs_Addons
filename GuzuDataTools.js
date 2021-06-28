//Ver 0.1.147c
//idea: atob() and btoa() //from window. encode decode base64

if(!window._guzuTF)window._guzuTF={};

class GuzuFileTools {
    /*
    inputId="";
    input=null;//file list
    files=null;//file indexer
    text=null;
    img=null; //an image to reuse to save memory
    onload=null;//general purpose
    indexPrefixFunction=null;
    */
    //function(){console.warn("GuzuDataTools: onload is not set");}
    
    setInputId(inputId_){
        this.inputId=inputId_;
        this.files=this.input=null;//re-initialize
        /*
        var src_=document.getElementById(inputId_);
        if (!src_)
            window.setTimeout(()=>{this.__link__()},1);
        else this.__link__(this);
        */
       return this;
    }
    
    /*
    __link__(aaa){
        var src_=document.getElementById(this.inputId);
        console.log(this);
        if (src_){
            var me=this;
            src_.addEventListener('onload',()=>{alert("?");if(me.onload)me.onload()});
        }
        
    }
    */
   
    constructor (inputId_){
        this.inputId="";
        this.input=null;//file list
        this.files=null;//file indexer
        this.text=null;
        this.img=null; //an image to reuse to save memory
        this.onload=null;//general purpose 
        //this.indexPrefixFunction=null;
        this.setInputId(inputId_);
    }
    
    applyIndexing(){
        var f_=(this.input || (this.input=document.getElementById(this.inputId))).files;

        this.files={};
        for (var i=0;i<f_.length;i++)
            this.files[i]=this.files[f_[i].name]=f_[i];
        return this.files;
    }
    
    
    //get a file by index or name
    file(id_){
        var f_=(this.input || (this.input=document.getElementById(this.inputId))).files;
        if (this.files!==null)return this.files[id_];
        if (typeof(id_)==='number')
            return f_[id_];
        //index if no index available
        return this.applyIndexing()[id_];
        
        /*for (var i=0;i<f_.files.length;i++) if (f_[i].name===id_)return f_[i];*/
    }
    
    //callback file as text. Returns false if failed
    getText(id_,callback_){ var me=this;
        var f_=this.file(id_);
        if (f_===undefined)return false;
        f_.text()
                .then((txt_)=>
            {
                me.text=txt_;
                if(callback_)
                    callback_({data:txt_});
            });
        return true;
    }
    
    //get a 2D array of {file:"filename",type:"image",options} and replaces the object by image data
    objectsToFiles(j_){
        var i,j;var me=this;
        
        for (i=0;i<j_.length;i++){
            for (j=0;j<j_.length;j++){
                //find objects to convert to files
                //format should be {file:"filename",type:"image",options}
                var f_=j_[i][j];
                if (typeof(f_)==='object'){

                    switch(f_.type){
                        case "image":
                            var o={i:i,j:j,f:f_,fn:function(o,e){
                               //console.log(o); 
                               j_[o.i][o.j]=e.data;
                               //o.f["data"]=e.data;
                            }};
                            me.getImagePixels(f_.file,o,j_[i][j].options);
                            break;
                        default:
                            console.warn("File type specified not supported.");
                            break;
                    }

                }//if object
            }
        };
    }

    //1/5/2021 TODO:testing
    //obj:array of file names or object with {file:"name.jpg"} which will be {file:"name",imageData:[]}
    //_options:{} will be sent to getImagePixels
    //return {ready:false,onReady:undefined,data:[] or {} } and you can set onReady=callback(obj,options)
    parseImages(obj,_options){
        var result={ready:false,onReady:undefined,data:Array.isArray(obj)?[]:{}};
        var replace_=(typeof(obj[0])!='object');//all should be consistant
        var total=0;
        var func=function(img,i){
            console.log(arguments)
            //i=i.i;
            console.log(i);
            if(replace_)
                obj[i]=img;
            else
                obj[i].imageData=img;
            result.data[i]=img;
            if(--total===0){
                result.ready=true;
                if(result.onReady)
                    result.onReady(obj,_options);
            }
            result.total=total;
            
        };
        for (var i in obj){
            var name=replace_?obj[i]:obj[i].file;
            var options=_options||(replace_?{}:obj[i].options||{});
            var vals={i:i}
            {this.getImagePixels(name,{i:i,fn:function(fn_,img){func(img,this.i)}},options);}
            total++;
        };
        return result;
    }

    //Obsolete: 1/5/2021
    //parse a JSON file to make use of its data for neural network
    NNParse(id_,callback_){
        var me=this;
        
        //=>
        if (!this.getText(id_,function(e){
            var j_=JSON.parse(e.data);
            //TODO:groups
            
            //convert objects to files
            me.objectsToFiles(j_.input);
            me.objectsToFiles(j_.output);
            //finally
            if(callback_)
                callback_({data:j_});
        }))return false;
        return true;
    }
    
    
    //splitter helps make training, validation and testing data
    NNSplitData(data,splitsArray=[8,2]){
        var i,j;
        //prepare empty arrays
        var s_=[];for ( i=0;i<splitsArray.length;i++)s_[i]=[];
        //make splits as regions
        for (j=1;j<splitsArray.length;j++)
            splitsArray[j]+=splitsArray[j-1];
        var max=splitsArray[splitsArray.length-1];
        for (i=0;i<data.length;i++){
            for (j=0;j<splitsArray.length;j++)
                if (i%max<splitsArray[j]){
                    s_[j].push(data[i]);break;
                }
        }
        return s_;
    }
    //TODO: maybe rename to setImagePixels
    //options {c:channels,m:multiply,a:add}
    NNSetImage(sourceData,targetImageData,options={}){
        //flatten arrays
        while (typeof(sourceData[0])==='object')
            sourceData=sourceData.flat();
        
        if (options.c===undefined)options.c=15;
        if (options.m===undefined)options.m=1;
        if (options.a===undefined)options.a=0;
        
        if(options.divide)options.m=options.divide;//does inverted action

        var mx =0;
        mx+=(options.c&1)===0?0:1;
        mx+=(options.c&2)===0?0:1;
        mx+=(options.c&4)===0?0:1;
        mx+=(options.c&8)===0?0:1;
        //console.log(options);
        var i,j,tmp;
        var imd=targetImageData;
        //if Canvas provided, use it
        if(targetImageData.nodeName){
            imd=imd.getContext('2d').getImageData(
                options.x|| 0,options.y|| 0,
                options.w || imd.width,
                options.h || imd.height
            );
        }
        var o=sourceData;
        for (i=0;i<o.length/mx;i++){
            j=0;
            if ((options.c)&1){
            tmp=imd.data[i*4+0]=((options.c&1)===0)?0:
                o[i*mx+j++]*options.m+options.a;
            
            }
            imd.data[i*4+1]=tmp=((options.c&2)===0)?0:
                    o[i*mx+j++]*options.m+options.a;
            imd.data[i*4+2]=tmp=((options.c&4)===0)?0:
                    o[i*mx+j++]*options.m+options.a;
            tmp=imd.data[i*4+3]=tmp=((options.c&8)===0)?255:
                    o[i*mx+j++]*options.m+options.a;
        };
      
      //if canvas provided, use it
      if(targetImageData.nodeName)
        targetImageData.getContext('2d').putImageData(imd,options.x|| 0,options.y|| 0);
      
    }
    
    
    //TODO: test batchBounds
    //fn_(a file) should return an array to append the data
    
    NNDatasetFromImages(options_,fn_,batchBounds){
        var result_={
            data:[[]],ready:false,readyCounter:0
        };
        
        var ready_=0;
        var f_=(this.input || (this.input=document.getElementById(this.inputId))).files;
        if (batchBounds===undefined)
            batchBounds=[0,f_.length];
        var _fn=function(callback_,o){
                   
                    
                    //o=o.data;
                    var i=this.i;
                    result_.readyCounter++;
                    result_.ready=(ready_===result_.readyCounter);
                    if (o.error)return ;
                    if (o.data!==undefined){
                        result_.data[0].push(o.data);
                        //console.log(o);
                        
                        if (fn_){console.log("o");
                            var data_=fn_(f_[i]);
                            //make sure to have enough columns
                            while(result_.data.length<=data_.length)
                                result_.data.push([]);
                            for (var j=0;j<data_.length;j++)
                                result_.data[j+1].push(data_[j]);


                        }
                            
                        
                    }
                    //ready_+","+result_.readyCounter);
                    
                    if (result_.ready && callback_.me.onload!=null)
                        callback_.me.onload(result_);
                }
        var _max=batchBounds[1];if (_max<0)_max=f_.length;//double check
        for (var i=batchBounds[0];f_[i]!==undefined && i<_max;i++){
            
            if(this.getImagePixels(i,{fn:_fn,i:i,me:this},options_))ready_++;
        }
        return result_;
    }
    
    //returns a new image object to use to draw
    getImage(id_,callback_,makeNewImage_=true){
        var img_;
        if (makeNewImage_)
            img_=new Image();
        else{
            if(img===null)img=new Image();
            img_=img;
        }
        var f_=this.file(id_);
        if (f_===undefined){console.warn("File not found");return null;}
        img_.src=URL.createObjectURL(f_);
        
        if (callback_)
            img_.onload=callback_;
        return img_;
    }
    
    __canvas__=null;
    //callback gets pixels as {data:}
    //callback is called if it is a function. If it is an object,
    //         callback.fn is called instead as fn(callback_,{data:r})
    //overwrite: if false, a clone of options will be used
    //returns true if file is an image
    getImagePixels(id_,callback_,options_,overwrite_=false){
        var i;
        //clone options to avoid overwrite
        if(!overwrite_)options_=Object.assign({},options_);
        //get File
        var img_=this.getImage(id_);
        var me=this;
        if(img_===null)return false;
        img_.onerror=(e)=>{
            if (callback_!==undefined)
                if(typeof(callback_)==='object')
                    callback_.fn(callback_,{error:e}); 
                else callback_({error:e});
        }
        img_.onload=()=>{
        
         //prepare options
            var opdefaults={
                 c:15//channels (r1 g2 b4 a8
                 //crop offset
                ,x:0,y:0
                ,r:0            //rotation
                ,w:img_.width   //*(options_.xscale||1)
                ,h:img_.height  //*(options_.yscale||1)
                ,xscale:1,yscale:1
                ,canvas:null
                ,rescale:!true   //rescale cropped area
                ,divide:1,add:0 //normalization tools
                ,func:null  //function to edit the results
            };

            if (!options_)
                options_={};

            for (i in opdefaults)
                if (options_[i]===undefined)
                    options_[i]=opdefaults[i];
            
            if (options_.rescale===true){
                options_.w*=options_.xscale;
                options_.h*=options_.yscale;
            }
           
         //prepare canvas
            var C,c2d;
            if (!options_.canvas){
                if (me.__canvas__===null){
                    me.__canvas__=document.createElement('canvas');
                    
                }
            }else{
                me.__canvas__=document.getElementById(options_.canvas);
            }
            //console.log( me.__canvas__)




            C=me.__canvas__;c2d=C.getContext('2d');
            C.width=options_.w;
            C.height=options_.h;
            var t_;
            
            if (options_.r!==0){
                options_.r *= Math.PI / 180;
                t_=0.5-0.5*Math.cos(options_.r);
                console.log("rotate");
                //c2d.translate(t_*img_.width,(-1+0.5*Math.sin(options_.r))*img_.height);
                c2d.translate(C.width*0.5,C.height*0.5);
                c2d.rotate(options_.r);
                c2d.translate(-C.width*0.5,-C.height*0.5);
               
            }
            
            c2d.drawImage(img_,options_.x,options_.y,img_.width*options_.xscale,img_.height*options_.yscale);
            if (options_.r!==0){
                c2d.rotate(-options_.r);
            }
            
            
            
            //if callback, send data to callback_
            if (callback_){
                var c_=options_.c;
                //convert string to number
                if (typeof(c_)==='string'){
                    c_= (c_.search("r")!==-1?1:0)+
                                (c_.search("g")!==-1?2:0)+
                                (c_.search("b")!==-1?4:0)+
                                (c_.search("a")!==-1?8:0);
                }
                c_=options_.c;
                
                //TODO
                var d_=(c2d.getImageData(0,0,C.width,C.height)).data;
                
                var i,j,k;var r=[];
                for (j=0;j<C.height;j++){
                    r[j]=[];
                    for (i=0;i<C.width;i++){
                        
                        r[j][i]=[];
                        k=j*C.width*4+i*4; 
                        if ((c_&1)!==0)r[j][i].push(d_[k  ]/options_.divide+options_.add);
                        if ((c_&2)!==0)r[j][i].push(d_[k+1]/options_.divide+options_.add);
                        if ((c_&4)!==0)r[j][i].push(d_[k+2]/options_.divide+options_.add);
                        if ((c_&8)!==0)r[j][i].push(d_[k+3]/options_.divide+options_.add);
                        //if (i==150 && j==50)alert(c_&1);
                        if (options_.func)r[j]=options_.func(r[j]);
                    }
                }
                
                //callback results
                if (typeof(callback_)==='object')
                    callback_.fn(callback_,{data:r});
                    else
                    callback_({data:r});
                        
            }
            
           
        };
        return true;
        
    }
    
    //TODO:further testing
    drawPixelsToCanvas(pixels,canvas,options_){
        if(typeof(canvas)==='string')
            canvas=document.getElementById(canvas);
        var c=canvas.getContext('2d');
        //default
        options_=options_||{};
        var defOptions={x:0,y:0,w:canvas.width,h:canvas.height,divide:1,add:0,func:null};//func need to be inverted
        for (var i in defOptions) if(!options_[i])options_[i]=defOptions[i];
        var o=options_;
        var c;
        var c2=(c).getImageData(o.x,o.y,o.w,o.h);
        if(!o.c)o.c=2**pixels[0][0].length-1;
        pixels=(pixels).flat().flat();
        for(var i in pixels){
            if(options_.func)
                pixels[i]=options_.func(pixels[i]);
            pixels[i]=(pixels[i]-options_.add)*options_.divide;
            
        }
        var flat=pixels;//.flat().flat();
        this.NNSetImage(flat,c2,{w:o.w,h:o.h,c:o.c,});
        c.putImageData(c2,0,0);
        return flat;
    }

    //returns a part of the array src_ looped
    sliceData(src_,iteration_,size_,loop_=true,stepSize_){
        if (stepSize_===undefined)stepSize_=size_;
        var from_=(iteration_*stepSize_)%src_.length;
        var re_=src_.slice(from_,from_+size_);
        var ex_=size_-re_.length;

        if (ex_>0 && loop_){
            return re_.concat(src_.slice(0,ex_));
        }
        return re_;
    }
    
    ModelgetWeights(model,toString=true){
        var i,w=[];
        for (i=0;i<model.weights.length;i++)
            w[i]=model.weights[i].read().dataSync();
            console.log(i)
        return toString?JSON.stringify(w):w;
    }

    ModelSetWeights(model,weights,fromString=true){
        var i,w=fromString?JSON.parse(weights):weights;
        var tf=window.tf;
        for (i=0;i<w.length;i++){
            w[i]=tf.tensor ( Object.values(w[i]) ,model.weights[i].shape);
            model.weights[i].write(w[i]);
        }
        

    }

}


