//ver 0.9b
if(!window._guzuTF)window._guzuTF={};
/*
class fixedDepthwiseConv2d extends tf.layers.Layer{
    static get className() {
        return 'fixedDepthwiseConv2d';
    }

    constructor(args) {
        super({});
        args=args||{};
        this.xd=args.xdim;this.yd=args.ydim;this.withr=args.withr;
        this.supportsMasking = true;
    }
    
    static get className() {
        return 'fixedDepthwiseConv2d';
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], inputShape[2], inputShape[3]];
    }

    call(it_, kwargs){
    }
}

tf.serialization.registerClass(fixedDepthwiseConv2d);
*/

tf.layers.effect=class effectLayers {

    
    /*
    * @param {Number} kwargs.normalize Make sure the maximum number doesn't exceed this
    * @param {Number} kwargs.magnify Multiplies the final result afterwards
    * @param {Number} kwargs.fade The further away, exponential decay for blur, multiply minus for edge
    
    
    */
    static blur(kwargs={}){
        if(kwargs.trainable===undefined)kwargs.trainable=false;
      
        if(kwargs.useBias===undefined)kwargs.useBias=false;
        if(typeof(kwargs.useBias)!='boolean'){kwargs.biasInitializer=kwargs.useBias;kwargs.useBias=true;}
      
        if(!kwargs.kernelSize)kwargs.kernelSize=[2,2];
        if(!kwargs.strides)kwargs.strides=[1,1];
        if(!kwargs.padding)kwargs.padding='same';

        if(!isNaN(kwargs.biasInitializer))//if the initializer is a number, use it
          kwargs.biasInitializer=tf.initializers.constant({value:kwargs.biasInitializer});      
        kwargs.depthwiseInitializer= tf.initializers.fixed({
            normalize:kwargs.normalize,
            fade:kwargs.fade,
            magnify:kwargs.magnify,
            type:'blur',
        });
        return tf.layers.depthwiseConv2d(kwargs);
    }

    static edge(kwargs={}){
        if(kwargs.trainable===undefined)kwargs.trainable=false;
        //
        if(kwargs.useBias===undefined)kwargs.useBias=true;
        if(typeof(kwargs.useBias)!='boolean'){kwargs.biasInitializer=kwargs.useBias;kwargs.useBias=true;}
        if(!kwargs.kernelSize)kwargs.kernelSize=[2,2];
        if(!kwargs.strides)kwargs.strides=[1,1];
        if(!kwargs.padding)kwargs.padding='same';
        
        if(kwargs.biasInitializer===undefined)kwargs.biasInitializer=tf.initializers.constant({value:.5});
        else
        if(!isNaN(kwargs.biasInitializer))//if the initializer is a number, use it
          kwargs.biasInitializer=tf.initializers.constant({value:kwargs.biasInitializer});
        kwargs.depthwiseInitializer= tf.initializers.fixed({
            normalize:kwargs.normalize,
            fade:kwargs.fade,
            magnify:kwargs.magnify,
            type:'edge',
        });
        return tf.layers.depthwiseConv2d(kwargs);
    }
  
  //TODO:needs testing. Replaces padding
    static border(inputLayer_){
      return {
        input:inputLayer_,
        apply:function(target){
          if (!Array.isArray(target))target=[target];
          //console.log("++",inputLayer_);
          return tf.layers.effect.padding(target[1] || this.input  ,  target[0]).apply(target[0]);
        }
      }
    }
    //requires applied layers
    static padding(inputLayer_,lastLayer_){
        var bb=lastLayer_;
        var b0=[inputLayer_.shape[1],inputLayer_.shape[2]];
        var s_=[(b0[0]-bb.shape[1])*.5,(b0[1]-bb.shape[2])*.5];
        return tf.layers.zeroPadding2d({padding:[[s_[0],s_[0]],[s_[1],s_[1]]]});
    }
}



window._guzuTF.fixedInitializer=class fixedInitializer extends tf.serialization.Serializable{
    
    static className = 'FixedInitializer';
    static config={normalize:1,fade:1,magnify:1,type:'blur'};
    className='FixedInitializer';
    //config={normalize:1};//value:this.value};
    
    getConfig() {
        return {
            //value: 1,//this.value,
            normalize:this.normalize,//zero no normalization. 1 default, other magnifies
            fade:this.fade,
            magnify:this.magnify,
            type:this.type
            };
        }
    
    //normalize
    apply(shape, dtype) {
        //console.log("TYPE    ",this.type);
        var effect_=this.type=='edge'?'toEdge':'toGBlur';
        //console.log("EFFECT:",effect_)
        return tf.tensor( this[effect_](shape,this.magnify,dtype));//tf.tensor( this.toGBlur(shape,1,dtype));
      }

    //value=0;
    constructor(args) {
        
        super();
        if (args.normalize===undefined)args.normalize=1;
        if (args.fade===undefined)args.fade=1;
        if (args.magnify===undefined)args.magnify=1;
        if (args.type===undefined)args.type='blur';
        var cc=this.config={};
        if (typeof args !== 'object') {
            throw new ValueError(
                `Expected argument of type ConstantConfig but got ${args}`);
        }
        if (args.value === undefined && false) {
            throw new ValueError(`config must have value set but got ${args}`);
        }
        //console.log("args",args);
        //this.normalize=args.normalize;
        //this.value = args.value;
        this.normalize=cc.normalize=args.normalize;
        this.fade=cc.fade=args.fade;
        this.magnify=cc.magnify=args.magnify;
        this.type=cc.type=args.type;

    }





    toGBlur(shape,gPower=1,dtype){
        var i,j,k;
        var arr_=[];
        var I=shape[0];
        var J=shape[1];
        var K=shape[2];
        var im=(I+1)*.5;
        var jm=(J+1)*.5;
        var total=0;var bb;
        //console.log("percentage:",pnt,"=",im,"/",jm);
        var pnt=im/jm;//percentage
        
        //var td=Math.sqrt(im*im+jm*jm);//total distance;
        var td=Math.sqrt((im*pnt)**2+(jm/pnt)**2);//total distance;
        var maxd=(im<jm?im:jm);
        //blur
        i=-1;while (++i<I){j=-1;arr_.push([]);
            while (++j<J){k=-1;arr_[i].push([]);
                //var d_=td-Math.sqrt((im-i)**2+(jm-j)**2)*this.fade;
                var d_=td-Math.sqrt((  (im-i-1)*pnt  )**2+(  (jm-j-1)/pnt )**2)*gPower;//*this.fade;
                //console.log("td",td);
                var vl=d_**this.fade;//gPower;//((I*J));
                while (++k<K)arr_[i][j].push([vl]);
                total+=vl;
                //console.log("vl",td,d_);
            }
        }
        //console.log("total",total);
        //console.log("normal:",this.normalize);
        //normalize
        if (this.normalize!=0){//console.log(" to normalize",this.normalize);
            if (total==0){
                total=1;//console.log("failed to normalize");
            }
            
            total=this.normalize/total;
            i=-1;while (++i<I){j=-1;
                while (++j<J){k=-1;
                    while (++k<K)arr_[i][j][k][0]*=total;
                }
            }
        }
        //console.log(arr_,dtype);
        //var result_=tf.tensor(arr_);
        //console.log()
        return arr_;//( result_ );
    }













    toEdge(shape,magnify=1,dtype){
        var i,j,k;
        var arr_=[];
        //console.log("shape",shape)
        var I=shape[0];
        var J=shape[1];
        var K=shape[2];
        var L=shape[3];
        var im=(I+1)*.5;
        var jm=(J+1)*.5;
        var total=0,total2=0;
        //console.log("percentage:",pnt,"=",im,"/",jm);
        var pnt=im/jm;//percentage
        
        //var magnify=this.normalize;
        var useAverage=this.normalize!=0;

        //var td=Math.sqrt(im*im+jm*jm);//total distance;
        var td=Math.sqrt((im*pnt)**2+(jm/pnt)**2);//total distance;
        var maxd=(im<jm?im:jm);

        i=0;while (++i<=I){j=0;arr_.push([]);
            while (++j<=J){k=-1;arr_[i-1].push([]);
                var d_=(this.fade==0)?1: 
                    td-Math.sqrt((  (im-i-1)*pnt  )**2+(  (jm-j-1)/pnt )**2)
                    *this.fade;
                //var vl=(i==im && j==jm)?0: ((((i/I+j/J)%1))<0.5?-1:1);
                var vl=-1;
                if (i<im && j<jm)vl=0;else
                if (i>=im && j>=jm)vl=2;
                    //( (i+im)/I+(j+jm)/J)//%2<1?-1:1;//(i+j)%2==0?d_:-d_;//d_**gPower;//((I*J));
                //vl=vl!=0?vl<0?-1:1:0;//TTT
                vl=vl*d_;
                while (++k<K){
                    arr_[i-1][j-1].push([]);
                    while(arr_[i-1][j-1][k].length<L)
                    arr_[i-1][j-1][k].push(vl);
                }
                if (i>=im && j>=jm)//TODO:should I remove abs?
                    total  +=Math.abs(vl);//*K?
                else total2+=Math.abs(vl);//*K?
            }
        }
        //console.log(arr_[0][1][0],arr_[1][1][0]);
        //console.log("tatoals:",total,total2);
        //console.log("toTest:");
        //console.log(arr_)
        total =this.normalize*magnify/total;
        total2=this.normalize*magnify/total2;
        i=-1;while (++i<I){j=-1;
            while (++j<J){k=-1;
                //var vl=Math.abs(im+jm-i-j)/((I*J));
                while (++k<K)arr_[i][j][k][0]*=useAverage?
                ((i>=im-1 && j>=jm-1)?total:total2)
                :magnify;
            }
        }

        //console.log("tatoals:",total,total2);
        //console.log(arr_[0][1][0],arr_[1][1][0])
        return arr_;
    }




}

tf.serialization.registerClass(window._guzuTF.fixedInitializer);
tf.initializers.fixed=(args)=>new window._guzuTF.fixedInitializer(args);
