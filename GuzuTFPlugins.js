//ver 1.5 TemporaryLayer modified
if(!window._guzuTF)window._guzuTF={};
window._guzuTF.AddCoords=class AddCoords extends tf.layers.Layer {
    //Idea from Uber
    static get className() {
        return 'AddCoords';
    }
    
    constructor(args) {
        super({});
        args=args||{};
        this.xd=args.xdim;this.yd=args.ydim;this.withr=args.withr;
        this.supportsMasking = true;
    }
    
    //static get className() {
    //return 'AddCoords';
    //}
    
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], inputShape[2], inputShape[3]/*+2*/];
    }
     
    call(it_, kwargs){
        var s=tf.scalar;
        var it=Array.isArray(it_)?it_[0]:it_;
        //if (this.g===undefined){console.log(it_);this.g=0;}
        var bs=it_[0].shape[0];
        this.xd=this.xd||it.shape[1];
        this.yd=this.yd||it.shape[2];
        this.invokeCallHook(it_, kwargs);//console.log(this.invokeCallHook);
        var xx_ones=tf.ones([bs,this.xd]).expandDims(-1);
        var xx_range=
                tf.range(0,this.yd).expandDims(0).tile([bs,1]).expandDims(1);
        var xx_channel = xx_ones.matMul(xx_range).expandDims(-1);
        
        var yy_ones=tf.ones([bs,this.yd]).expandDims(1);
        var yy_range=
                tf.range(0,this.yd).expandDims(0).tile([bs,1]).expandDims(-1);
        var yy_channel = yy_range.matMul(yy_ones).expandDims(-1);
        
        
        xx_channel=xx_channel.asType('float32').div( s(this.xd).sub(s(1))  );
        yy_channel=yy_channel.asType('float32').div( s(this.yd).sub(s(1))  );
        
        xx_channel=xx_channel.mul(tf.scalar(2)).sub(s(1));//.variable();
        yy_channel=yy_channel.mul(tf.scalar(2)).sub(s(1));//.variable();
        
        var ret= it_.concat([xx_channel,yy_channel],-1);
        //if (tmp_===undefined){console.log(it.arraySync());tmp_=ret;console.log(ret[0].arraySync());}
        if (this.withr){
            var rr=xx_channel.square().add(yy_channel.square()).sqrt();
            ret = ret.concat(rr);
        }
        
        return ret;
        
    }
}
//var tmp_;

tf.serialization.registerClass(window._guzuTF.AddCoords);  // Needed for serialization.
//export function guzuCoordConv() {return new GuzuCoordConv();}




//to use: new AddScalar({values:[6,7]})
window._guzuTF.AddScalar=class AddScalar extends tf.layers.Layer {
    
    static className='AddScalar';
    
    
    constructor(args) {
        super({});
        args=args||{};
        //this.xd=args.xdim;this.yd=args.ydim;this.withr=args.withr;
        this.supportsMasking = true;
        this.values=args.values;
    }
    
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1]+this.values.length/*+2*/];
    }
     
    call(it_, kwargs){
      this.invokeCallHook(it_, kwargs);
      this._dim=this.values.length;
      var res;//=it_;
      res=it_;
      
      res[0]=it_[0].concat(tf.tensor([this.values]).tile([it_[0].shape[0],1]),-1);
      
      
      return res;
    }
}

tf.serialization.registerClass(window._guzuTF.AddScalar);  // Needed for serialization.


//epic win
/**args:{

    weight: multiply the input by this before processing
    scale: multiply the output by this before sending
    slope: (default:100) steepness of error tolerance 
    useSum: (default:true) returns the sum of occurences. Otherwise, replace the found value with 1
    
    find:Array of values to find
 if find not used:
    range: default:[0,1] range in which the units are devided
    units: count of numbers to use in the range
   }
*/
window._guzuTF.AddCounter=class AddCounter extends tf.layers.Layer {
    
    //static className='AddCounter';
    static get className() {
        return 'AddCounter';
    }
    
    constructor(args) {
      args=args||{};
      args.trainable=args.trainable===undefined?args.type>0:args.trainable;
      super(args);
      
      //this.xd=args.xdim;this.yd=args.ydim;this.withr=args.withr;
      this.supportsMasking = true;
      //this.values=args.values;
      this.find=args.find;
      this.units=args.units;
      this.range=args.range;
      this.weight=args.weight||1;// should be |10
      this.scale=args.scale||1;
      this.slope=args.slope||100
      this.useSum=args.useSum===undefined?true:args.useSum;
      this.type=args.type!==undefined?args.type:1;

      this.step=args.step===undefined?.499:args.step;
      this.useSigmoid=args.useSigmoid!==false;
      this._createFindValues();
        
      //this.find=this.find.map(e=>[[e]]);
      this.find=this.find.map(e=> [[[e]]]);
     
    }
  
    _createFindValues(){
      //create values to find
      if(this.find===undefined){
        this.find=[];
        if(this.range===undefined)this.range=[0,1];
        if(this.units===undefined)this.units=10;
        var r=this.range;var u=this.units;var s=r[1]-r[0];
        for(var i=0;i<u;i++)
          this.find.push(i*s/u+r[0]);
      }
      this.units=this.find.length;
    }

    build(){//console.log(this.weight)
      if(this.type>0){
        this.weight_=this.addWeight('weight',[1],'float32',tf.initializers.constant({value:this.weight}));
        this.slope_=this.addWeight('slope',[1],'float32',tf.initializers.constant({value:this.slope}));
        this.scale_=this.addWeight('scale',[1],'float32',tf.initializers.constant({value:this.scale*2}));
        this.bias_=this.addWeight('bias',[this.units],'float32',tf.initializers.constant({value:1}));
      }
    }
  
    
    computeOutputShape(inputShape) {//TODO
      //var
      var outputshape;
      if(this.useSum){
        if(inputShape.length!=4)
          outputshape= [inputShape[0],this.find.length];
        else//use channel
            outputshape=[inputShape[0], inputShape[inputShape.length-1],this.units/*+2*/];
      }else{
        if(inputShape.length!=4)
          outputshape= [inputShape[0],inputShape[1]*this.find.length];
        else//use channel
            outputshape=[inputShape[0],inputShape[1],inputShape[2], inputShape[3]*this.units];
      }
      //console.log("SHAPE:",outputshape)
      return outputshape;
    }
     
    call(it_, kwargs){
      return tf.tidy(() =>  this.callTidy(it_, kwargs));
    }
    
    callTidy(it_, kwargs){
      this.invokeCallHook(it_, kwargs);
      var res,a;//=it_;
      res=it_;
      a=it_[0];
      var dim=res[0].shape.length;
      
      
      var prem;
      //if(false)
      switch(dim){
        case 4://make channel first and flatten the rest
          res[0]=res[0].transpose([0,3,1,2]).reshape([a.shape[0],a.shape[3],a.shape[1]*a.shape[2]]);
          prem=[1,0,2];
          break;
        case 3://TODO
          res[0]=res[0].reshape([a.shape[0],a.shape[1]*a.shape[2]]);
          console.log("Shape dim 3 not ready. //TODO: Testing");//res[0].print();
          break;
        case 2:
        //console.log("o0o")
          //console.log("o0o:2");
          break;
      }
      //res[0].print()
      
      res[0]=this.callType(this.type,it_,kwargs);
      
      if (this.useSum){
        res[0]=res[0].sum(-1);//.round();
        res[0]=res[0].transpose(prem);//res[0].print()
      }
      else{
        
        if(dim!=4){//WIP
          res[0]=res[0].transpose([2,1,0,3]);
          var r=res[0].shape;//console.log(r)
          res[0]=res[0].reshape([r[0],r[1]*r[2]*r[3]]);
        }
        else{
          var r=res[0].shape;//console.log(r)
          res[0]=res[0].transpose([1,0,2,3]);
          r=res[0].shape;//console.log(r,a.shape)
          res[0]=res[0].reshape([r[0],r[1]*r[2], a.shape[1],a.shape[2]  ]);
          res[0]=res[0].transpose([0,2,3,1]);
          r=res[0].shape;//console.log(r)
        }
      }
      return res[0];
       //this.dataFormat ==='channelsFirst'
    }

    callType(type,it_,kwargs){
      var res,uu; //result,temp var
      if(Array.isArray(it_))
        it_=it_[0];
      switch(type){
        case 0:
          res=it_.mul(this.weight)
                .sub( tf.tensor(this.find))
                .mul(this.slope)
                .pow(2).mul(tf.scalar(-1));
          res=this.callStep(res,{sigmoid:this.useSigmoid,step:this.step})
              .mul(this.scale);
          break;
        case 1:
          res=it_.mul(this.weight_.read())
                .sub( (uu=tf.tensor(this.find)).mul(this.bias_.read().reshape(uu.shape))  )
                .mul(this.slope_.read())
                .pow(2).mul(tf.scalar(-1));
          res=this.callStep(res,{sigmoid:this.useSigmoid,step:this.step})
              .mul(this.scale_.read());
          break;

        case 2:
          res=it_.mul(this.weight_.read())
                .sub( (uu=tf.tensor(this.find)).mul(this.bias_.read().reshape(uu.shape))  )
                .mul(this.slope_.read())
                .pow(2).mul(tf.scalar(-1));
          res=this.callStep(res,{sigmoid:this.useSigmoid,step:this.step})
              .mul(this.scale_.read());
          break;

        case -1://TODO: didn't start testing
        //console.log(this.slope_)
          res=it_.mul(this.weight).sub(tf.tensor(this.find)).pow(2);
          res=tf.scalar(1).div(res.mul(this.slope).add(.5)).mul(.5);
          //res=tf.scalar(1).div(res).sub(1);

          res=res.mul(this.scale);
          break;
      }
      return res;//
    }

    callStep(v,args){
      if(args.sigmoid)
        v=v.sigmoid();
      if(args.step!==undefined)
        v=v.sub(args.step).step();
      return v;
    }
}

      

tf.serialization.registerClass(window._guzuTF.AddCounter);  // Needed for serialization.


//TODO: Test Everything
window._guzuTF.SumPooling2d=class SumPooling2d extends tf.layers.Layer {
  static get className() {
        return 'SumPooling2d';
    }
  constructor(args) {
      args=args||{};
      args.trainable=false;
      super(args);
    this.strides=args.strides?(
      Array.isArray(args.strides)?args.strides:[args.strides,args.strides]):[1,1];
    this.padding=args.padding||'valid';
    this.poolSize=!args.poolSize?[1,1]:Array.isArray(args.poolSize)?args.poolSize:[args.poolSize,args.poolSize];
    
    this.__={};
    
  }
  
  computeOutputShape(inputShape) {
    switch(this.padding){
      case 'valid':
        var x=~~((inputShape[1]-this.poolSize[0])/this.strides[0]+1);
        var y=~~((inputShape[2]-this.poolSize[1])/this.strides[1]+1);
         //console.log("ee")
        return [inputShape[0],x,y,inputShape[3]];
      case 'same':
        return [inputShape[0],Math.ceil(inputShape[1]/this.strides[0]),Math.ceil(inputShape[2]/this.strides[1]),inputShape[3]];
      default:
      return inputShape;
    }
  }
  
  call(it, kwargs){ 
    it=Array.isArray(it)?it[0]:it;
    var ones=this.makeOnes(it.shape[3]);
    //ones.print();
    var out=it.depthwiseConv2d(ones,this.strides,this.padding);//console.log("S:",out.shape)
    return out;
  }//call
  
  makeOnes(channels_=1){
    return this.__.ones || (this.__.ones=tf.keep(tf.ones(
      this.poolSize.concat(channels_).concat(1)
    )));//[2,2,3,1]);
  }
  
}//SumPooling2D

tf.serialization.registerClass(window._guzuTF.SumPooling2d);  // Needed for serialization.

window._guzuTF.Mutation2dInfo={};
window._guzuTF.Mutation2d=class Mutation2d extends tf.layers.Layer{
  static get className() {
    return 'Mutation2d';
  }
  constructor(args) {
    args=args||{};
    args.trainable=false;
    super(args);
    var F=window._guzuTF.Mutation2dInfo;
    var f;
    //Record data if name is specified
    if(!args.set && !args.get)args.set="$$DEFAULT$$";
    this.set=args.set;//initiate setting
    this.get=args.get;//use previous setting
    this.flip=args.flip;//true false
    this.fill=args.fill||0;
    this.paused=args.paused||false;
    if (args.set){
      f=F[this.set]={};
      f.rotation=args.rotation||0;
      f.offset=args.offset||[0,0];
      if(!Array.isArray(f.offset))
        f.offset=[f.offset,f.offset];

      f.flip=args.flip||false;
      f._flip=false;

      var A=Array.isArray;

      f.c=args.channels;//[add,multiply,min,max]
      if(f.c){
        if( !A(f.c))
          f.c=[f.c];
        while(f.c.length<4)
          f.c.push([0,0,1][f.c.length-1]);
      }
    }//args.set


    //use recorded data
    //f=F[args.name||args.use];
    //for(var i in f)
    //  this[i]=f[i];

  }//constructor

  computeOutputShape(inputShape) {
    return inputShape;
  }

  call(it, kwargs){ 
    it=Array.isArray(it)?it[0]:it;
    //ones.print();
    if(this.paused)
      return it;
    var F=window._guzuTF.Mutation2dInfo;
    var f=F[this.set||this.get];
    if(this.set){
      //randomize
      //f=F[this.set];
      //rotation effect
      if(f.rotation){
        f.r=f.rotation*(Math.random()*2-1);
        f.x=f.offset[0]*(Math.random()*2-1)+0.5;
        f.y=f.offset[0]*(Math.random()*2-1)+0.5;
        //if f.fill is Array of 2 values, use random fill
        if (Array.isArray(f.fill) && f.fill.length==2)f.f=[
          Math.random()*(f.fill[1]-f.fill[0])+f.fill[0],
          Math.random()*(f.fill[1]-f.fill[0])+f.fill[0],
          Math.random()*(f.fill[1]-f.fill[0])+f.fill[0],
        ];
        else f.f=f.fill;
      }//rotation

      if(f.flip)
        f._flip=Math.random()>.5;

      if(f.c)
        f._c=[
          (Math.random()*2-1)*f.c[0],
          (Math.random()*2-1)*f.c[1]+1,
          f.c[2],
          f.c[3],
        ]

    }//this.set
    var out=it;
    if(f._c)
      out=tf.minimum(
        tf.onesLike(out).mul(f._c[3]),
        tf.maximum(out.mul(f._c[1]).add(f._c[0]),tf.onesLike(out).mul(f._c[2]))
      );
    if(f._flip)
      out=tf.image.flipLeftRight(out);
    if(f.rotation)
      out=tf.image.rotateWithOffset(out,f.r,f.f,[f.x,f.y]);//console.log("S:",out.shape)






      
    return out;
  }//call
}/////


tf.serialization.registerClass(window._guzuTF.Mutation2d);  // Needed for serialization.



//////
//example use: tf.layers.temp({call:(inp)=>inp.mul(2)})
window._guzuTF.TemporaryLayer=class TemporaryLayer extends tf.layers.Layer {
  static get className() {
        return 'Temporary';
    }
  constructor(args) {
      args=args||{};
      super(args);
      this.__={};
      this.__.this=this;
      for (var i in args){
        this.__[i]=args[i];
      }
      
    
  }
  
  build(){
    if(this.__.build)
       this.__.build(this.__);
  }

  computeOutputShape(inputShape) {
    if(this.__.computeOutputShape)
      return this.__.computeOutputShape(inputShape,this.__);
    else 
      return inputShape;
  }
  
  call(it, kwargs){ 
    it=Array.isArray(it)?it[0]:it;
    return this.__.call(it,kwargs,this.__);
  }//call
  
  
}//SumPooling2D

tf.serialization.registerClass(window._guzuTF.TemporaryLayer);  // Needed for serialization.

//input is a tensor2d or tensor3d
//returns a bouning box of shape [x1,x2,y1,y2]. x1 can contain all channels
//args:{normalize:false,threshold:0,useThreshold:false}
//normalize for range 0-1. Threshold to convert values to 1 if exceeded it
window._guzuTF.BoundingBoxLayer=class BoundingBoxLayer extends tf.layers.Layer {
  static get className() {
        return 'BoundingBoxLayer';
    }
  
  constructor(args) {
    //args={normalize:bool,threshold:Number,useThreshold:bool}
    args=args||{normalize:false,threshold:0,useThreshold:false};
    super(args);
    //this.__={};
    for (var i in args){
      this[i]=args[i];
    }
    
  }
  
  computeOutputShape(inputShape) {
    if(inputShape.length===3)
      return [inputShape[0],4];
    else
      return [inputShape[0],4,inputShape[3]];
  }
  
  call(it, kwargs){ 
    var a=Array.isArray(it)?it[0]:it;
    if(this.useThreshold)
      a=a.sub(this.threshold).step();
    var d=(a.shape.length===4)?1:0
    var s1=a.shape[a.shape.length-1-d];
    var s2=a.shape[a.shape.length-2-d];
    var b=tf.range(s1,0,-1)//.expandDims(1);
    b=d?b.expandDims(1):b;
    //b.print();
    var c,r;
    c=a.mul(b).max(-1-d).max(-1-d,true);
    var r=tf.scalar(s1+1).sub(c);//r.print();
//
    c=a.mul(b.reverse()).max(-1-d).max(-1-d,true);
    r=r.concat(c,-1-d);//r.print();
//
    b=tf.range(s2,0,-1).expandDims(1);
    c=tf.scalar(s2+1).sub(a.mul(d?b.expandDims(1):b).max(-1-d).max(-1-d,true));//c.print();
    r=r.concat(c,-1-d);//r.print();
//
    b=d?b.reverse().expandDims(1):b.reverse();
    c=a.mul(b).max(-1-d).max(-1-d,true);//c.print();
    r=r.concat(c,-1-d).sub(1);
    if(this.normalize){
      s1--;s2--;
      var ss=tf.tensor([1/s1,1/s2,1/s1,1/s2]);
      ss=d?ss.expandDims(1):ss;
      
      return r.mul(ss);
    }
    return r;
  }//call
  /*
  // Find Bounding box
var s=[]; for(j=0;j<2;j++)for(var i=0;i<6;i++)
  s=s.concat([s.length>4&&j==1?1:0, s.length>4?1:0, s.length>1?.5:0,  s.length>1?1:0,0,0])

var d=0;
var a=d?tf.tensor4d(s,[2,3,3,4]):tf.tensor3d(s,[2,6,6]);
a=a.step();a.print();
var s1=a.shape[a.shape.length-1-d];
var s2=a.shape[a.shape.length-2-d];

var b=tf.range(s1,0,-1);
b=d?b.expandDims(1):b;

//b.print();
var c;
c=a.mul(b).max(-1-d).max(-1-d,true);
var r=tf.scalar(s1+1).sub(c);//r.print();
;
c=a.mul(b.reverse()).max(-1-d).max(-1-d,true);
r=r.concat(c,-1-d);//r.print();
console.log("======================");

b=tf.range(s2,0,-1).expandDims(1);
c=tf.scalar(s2+1).sub(a.mul(d?b.expandDims(1):b).max(-1-d).max(-1-d,true));//c.print();
r=r.concat(c,-1-d);//r.print();

//c=a.mul(b.reverse());//c.print();
b=d?b.reverse().expandDims(1):b.reverse();
c=a.mul(b).max(-1-d).max(-1-d,true);//c.print();
s1--;s2--;
var ss=tf.tensor([1/s1,1/s2,1/s1,1/s2]);
ss=d?ss.expandDims(1):ss;
r=r.concat(c,-1-d).sub(1).print();//r.sub(1).mul(ss).print();
;
   */
  
}//BoundingBoxLayer

tf.serialization.registerClass(window._guzuTF.BoundingBoxLayer);  // Needed for serialization.





///tf.layers.weight1d
//apply([inputLayer,weightSourceLayer])
//  Make sure that weightSourceLayer size is a multiple of: inputLayer size + biasUnits arg below
//args:{biasUnits:number of bias units to add}
window._guzuTF.Weight1DLayer=class Weight1DLayer extends tf.layers.Layer {
  static get className() {
        return 'Weight1DLayer';
  }
  constructor(args) {
    //args={normalize:bool,threshold:Number,useThreshold:bool}
    args=args||{};//normalize:false,threshold:0,useThreshold:false};
    super(args);
    this.biasUnits=args.biasUnits||0;
    this.init=args.init??true;
  }
  computeOutputShape(inputShape) {

    var a=inputShape[0];
    var b=inputShape[1];

    //initialize
    if (this.init)
      if (b[1]<a[1]){
        console.error("Weight1DLayer: Input size is larger than the weights.");
        this.init=false;
      }
      else{
          
          

          this.init=false;
          var v=b[1]%(this.biasUnits+a[1]);
          //console.log("a1:",a[1]," v:",v," b1:",b[1]," bias:"+this.biasUnits)
          if (v>0){

            var i=1;
            //Try compensate bad size
            while(i<100 && b[1]%(this.biasUnits+a[1]+i)>0)
              i++;
            //if i = 100 then fail
            
            (i<100?console.warn:console.error)("Weight1DLayer: Second layer isn't a multiple of the first."+
            (i<100?" Adding "+i+" to biasUnits to compensate.":""));
            if (i<100)this.biasUnits+=i;
          }
      }

    var c=[a[0],b[1]/(a[1]+this.biasUnits)];
    //console.log(c);
    return c;
  }
  call(it, kwargs){ 
    var a=it[0];//Array.isArray(it)?it[0]:it;
    var b=it[1];
    //console.log("bu:"+this.biasUnits)
    if(this.biasUnits>0){
      var o=tf.ones([a.shape[0],this.biasUnits]);//o.print();//adding bias
      a=tf.concat([a,o],1);
    }


    var targetShape=[a.shape[0],a.shape[1],b.shape[1]/a.shape[1]];
    //console.log("target Shape:",targetShape,"oldShape:",b.shape,"a shape:",a.shape)
    b=b.reshape(targetShape);
    a=a.expandDims(-1);
    
    a=a.mul(b).sum(-2);

    //console.log(a.shape);
    return a;
  }
}

tf.serialization.registerClass(window._guzuTF.Weight1DLayer);  // Needed for serialization.
















//{//Todo: WIP


///tf.layers.convWeight1d
//apply([inputLayer,weightSourceLayer])
//  Make sure that weightSourceLayer size is a multiple of: inputLayer channel size * kernel size * #of expected filters
//args:{biasUnits:number of bias units to add} //TODO
window._guzuTF.ConvWeight2DLayer=class ConvWeight2DLayer extends tf.layers.Layer {
  static get className() {
        return 'ConvWeight2DLayer';
  }
  constructor(args) {
    //args={normalize:bool,threshold:Number,useThreshold:bool}
    args=args||{};//normalize:false,threshold:0,useThreshold:false};
    super(args);
    this.biasUnits=args.biasUnits||0;
    this.init=args.init??true;
    this.kernelSize=args.kernelSize??[1,1];
    if(!isNaN(this.kernelSize))
      this.kernelSize=[this.kernelSize,this.kernelSize];
    this.strides=args.strides??[1,1];
    if(!isNaN(this.strides))
      this.strides=[this.strides,this.strides];
    this.padding=args.padding??'same';
    this.tensor=args.tensor;
    this.size=args.size;
    this.seed=args.seed??0;
    this.noise=args.noise??0;
    this.cosGain=args.cosGain??1;
    this.layerBased=!this.tensor && !this.size;
    
    //console.log("LB:"+this.layerBased)
    
  }

  makeTensor(){
    var a;
    var x=this.size[0],y=this.size[1];
    //a=tf.range(0,y).div(y-1). expandDims(1).tile([1,x]). stack( tf.range(0,x).div(x-1) .expandDims(0).tile([y,1]),2)
    a=tf.range(0,y).div(y-1). expandDims(1).tile([1,x]).expandDims(2);//. stack( tf.range(0,x).div(x-1) .expandDims(0).tile([y,1]),2)
    for (i=1;i<this.size[2]??2;i++)
      a=a.concat((i&1)
      ?this._cos(tf.range(0,x).div(x-1),i) .expandDims(0).tile([y,1]).expandDims(2)
      :this._cos(tf.range(0,y).div(y-1),i). expandDims(1).tile([1,x]).expandDims(2)
      ,2)
    
    return this.tensor=a;
  }

  _cos(t,i){
    window.ttt=this;
    if (i<2 && i>-1)return t;
    var I=(i%4<2)?1:-1;
    i=0|(i*.5);//To compensate for x and y
    
    return tf.cos(t.mul(Math.PI*i*this.cosGain)  ).mul(I);

  }


  computeOutputShape(inputShape) {
    var a,b;//console.log(inputShape)
    if (this.layerBased){
      a=inputShape[0];
      b=inputShape[1];
    }else{
      b=inputShape;//[0];
      if (this.tensor)
        a=[b[0]].concat(this.tensor.shape);
      else
        a=[b[0],this.size[0],this.size[1],this.size[2]??2];
    }
    

    //initialize and remember output shape
    if (this.init){this.init=false;
      if (false){
        if (b[1]<a[1]){
          console.error("ConvWeight2DLayer: Input size is larger than the weights.");
          
        }
        else{
            var v=b[1]%(this.biasUnits+a[1]);
            //console.log("a1:",a[1]," v:",v," b1:",b[1]," bias:"+this.biasUnits)
            if (v>0){

              var i=1;
              //Try compensate bad size
              while(i<100 && b[1]%(this.biasUnits+a[1]+i)>0)
                i++;
              //if i = 100 then fail
              
              (i<100?console.warn:console.error)("ConvWeight2DLayer: Second layer isn't a multiple of the first."+
              (i<100?" Adding "+i+" to biasUnits to compensate.":""));
              if (i<100)this.biasUnits+=i;
            }
        }
      }

      //console.log("a",a,"b",b,"in",inputShape,"lb:",this.layerBased);
      var targetShape=[ this.kernelSize[0],this.kernelSize[1],a[3]
      ,(b[1]/ (a[3]*this.kernelSize[0]*this.kernelSize[1]))    ];
      if (targetShape[3]!=(targetShape[3]|0))
        throw(`ConvWeight2D Error: Resulting shape has fractions.\n"${b[1]}/(${a[3]}*${targetShape[0]}*${targetShape[1]})=${targetShape[3]}" on layer ${this.name}.`
        +` \nPlease check: (filter size)/(input channels* kernelSize)`);
      //console.log((targetShape[3]!=(targetShape[3]|0)))
      //console.log(a,b,this.kernelSize,targetShape);
      this.outputshape_=tf.conv2d(tf.ones(a.slice(1)),tf.ones(targetShape),this.strides,this.padding).shape;
      
    }

    var c=this.outputshape_;
    c=[a[0],c[0],c[1],c[2]];
    //console.log("outputShape:",c);
    //var c=[a[0],b[1]/(a[1]+this.biasUnits)];
    //console.log(c);
    return  c;
  }


  build(){
    if(!this.tensor && this.tensor!==false)
      this.tensor=this.makeTensor();
    if (this.noise)
      this.tensor=this.tensor.add(tf.randomUniform(this.tensor.shape,-this.noise,this.noise,'float32',this.seed??0));
    //this.tensor=this.tensor;
  }

  call(it, kwargs){ 
    var a,b;
    var i,j,result;
    var length;
    var layerBased_=this.layerBased;//if two layers used instead of one tensor
    if(Array.isArray(it)){
      a=it[0];//Array.isArray(it)?it[0]:it;
      b=it[1];
    }else 
      a=it;
    length=a.shape[0];
    //
    //console.log("it:",it,it[0].shape)
    if (!b){
      layerBased_=false;
      b=a;
      //if(this.tensor)
      a=this.tensor;//else a=this.makeTensor();
    }


    
    //console.log("bu:"+it[0])
    
    if(false && this.biasUnits>0){
      //var o=tf.ones([a.shape[0],this.biasUnits]);//o.print();//adding bias
      //a=tf.concat([a,o],1);
    }
    //console.log("a shape:"+a.shape,"b shape:"+b.shape,"b size:"+b.size)
    var a3=layerBased_?a.shape[3]:a.shape[2];
    var targetShape=[length,this.kernelSize[0],this.kernelSize[1],a3
      ,(b.size/(length*a3*this.kernelSize[0]*this.kernelSize[1])  )];//a.shape[1],b.shape[1]/a.shape[1]];
    //console.log(`ConvWeight2D:\n"${b.size}/(${a3}*${targetShape[1]}*${targetShape[2]})=${targetShape[4]}" on layer ${this.name}.`)
    //var targetShape=[a.shape[0]].concat(this.outputshape_);
    //console.log("target Shape:",targetShape,"oldShape:",b.shape,"a shape:",a.shape)
    b=b.reshape(targetShape);
    //a=a.expandDims(-1);
    //a=a.mul(b).sum(-2);
    //console.log("b shape",b.shape);
    
    
    b=length>1?b.split(length,0):[b];
    if(layerBased_)
      a=length>1?a.split(length,0):[a];
      
    //console.log("a new shape ",a)
    //console.log("layerb",layerBased_)
    for (i=0;i<length;i++){//if(i<2)console.log("i=",i,length,a.shape);//b[i].print()//.slice([i],[1]).print();
      j=tf.conv2d(layerBased_?a[i]:a,b[i].squeeze(0),this.strides,this.padding);
      //j=tf.conv2d(a.slice([0],[1]),b.slice([0],[1]).squeeze(0),this.strides,this.padding);
      //console.log("i=",i)
      /*
      if(i==0)
        result=j.expandDims(0);
      else
        result=result.concat(j.expandDims(0),0);
        */
      result=i?result.concat(j.expandDims(0),0):j.expandDims(0);
      //window.J=result; 
      //result.sum().print();
      //console.log(a.shape);
    }
    
    
    if(result.shape.length>4)
      result=result.squeeze(1);
    //console.log("result shape:",result.shape)
    //result.print();
    //if(result[0]==1)result.squeeze();

    return result.clone();
  }
}

tf.serialization.registerClass(window._guzuTF.ConvWeight2DLayer);  // Needed for serialization.



//}Todo





































//args:{index:[numbers]}
//returns input with channels removed
//TODO: full testing
window._guzuTF.DropChannelsLayer=class DropChannelsLayer extends tf.layers.Layer {
  static get className() {
        return 'DropChannelsLayer';
    }
  
  constructor(args) {
    //args={normalize:bool,threshold:Number,useThreshold:bool}
    args=args||{index:[]};//normalize:false,threshold:0,useThreshold:false};
    super(args);
    args.index.sort();
    //this.__={};
    for (var i in args){
      this[i]=args[i];
    }
    
  }
  
  computeOutputShape(inputShape) {
    inputShape[inputShape.length-1]-=this.index.length;//no need to concat()? O_O;
    return inputShape;
  }
  
  call(it, kwargs){ 
    var a=Array.isArray(it)?it[0]:it;
    var c=it.shape[it.shape.length-1];

    if(this.index.length===0)//if no indicies provided, do nothing
      return a;
    //drop starts
    var n=this.index.concat();
    a=a.split(c,-1);
    var b;
    i=-1;while(++i<c){
      if(i!==n[0])
        b=b?b.concat(a[i],-1):a[i];
      else
        n.shift();
    }

    //if all channels dropped, show error
    if(!b)
      console.error("dropChannel returned no channels!");
    return b;
  }
  
}//DropChannelsLayer

tf.serialization.registerClass(window._guzuTF.DropChannelsLayer);  // Needed for serialization.





/*
 * Accepts both ranges and two arrays of ranges
*/
window._guzuTF.GuzuTfTools=class GuzuTfTools{
  layerMapper(applyto,low1,high1,low2,high2){
      
      if (Array.isArray(low1)){
        low2=high1[0];high2=high1[1];
        high1=low1[1];low1=low1[0];
      }
      applyto=tf.layers.reshape({targetShape:[applyto.shape[1],1]}).apply(applyto);
      applyto=tf.layers.conv1d({kernelSize:[1],filters:1,trainable:false,
                          kernelInitializer:tf.initializers.constant({value:1}),
                          biasInitializer:tf.initializers.constant({value:-low1})
                         }).apply(applyto);
      applyto=tf.layers.conv1d({kernelSize:[1],filters:1,trainable:false,
                          kernelInitializer:tf.initializers.constant({value:(low2-high2)/(low1-high1)}),
                          biasInitializer:tf.initializers.constant({value:+low2})
                         }).apply(applyto);
      return tf.layers.flatten().apply(applyto);
    }
  //applyto didn't use apply yet. Others should have already been applied
  layerPass(applyto,topass,toblock){
    var c1=applyto.apply( tf.layers.concatenate().apply([topass,toblock]) );
    return tf.layers.concatenate().apply([c1,topass]);
    
  }
  
  map(low1,high1,low2,high2){
    var t=this;
    console.log("--",t)
    var f=(applyto)=>{//console.log(arguments)
      return t.layerMapper(applyto,low1,high1,low2,high2);
    };
    return{
      apply:f
    }
  }
  
  pass(topass,toblock){
    var t=this;
    return{
      apply:function(applyto){
        return t.layerPass(applyto,topass,toblock);
      }
    }
  }
  
  
  
  //multiplies the previous layer by constant
  mul(val,bias){
    var m={
      v:val===undefined?-1:val,
      b:bias?bias:0,
      apply:function(applyto){
        var tool="conv"+((applyto.rank-2)||1)+"d";
        //console.log(applyto);
        //console.log(tool);
        //if (applyto.shape)
        var result=tf.layers[tool]({kernelSize:1,filters:applyto.shape[applyto.shape.length-1],
                                 trainable:false,
                          kernelInitializer:tf.initializers.constant({value:this.v}),
                          biasInitializer:tf.initializers.constant({value:this.b})
                         });
        result.name=result.name.replace('conv','multiply')
        return result.apply(applyto);
      }
    }
    
    return m;
  }
  
  sub(bias){
    var t=this;
    return {
      apply:function(applyto){
        var a1=t.mul(-1,bias).apply(applyto[1]);
        var add=tf.layers.add();
        add.name=add.name.replace('add','subtract')
        //add.name=add.name.replace('Add','Subtract')
        //console.log("SUB",add)
        a1=add.apply([applyto[0],a1]);
        
        
        return a1;
      }
    };
  }
  
  
  
}
window._guzuTF.guzuTfTools=new window._guzuTF.GuzuTfTools();

//tf.guzu.tools
tf.guzu={};
{
  
  tf.guzu.removeChannels=function(it,index=[]){
    var a=Array.isArray(it)?it[0]:it;
    var c=it.shape[it.shape.length-1];

    if(index.length===0)//if no indicies provided, do nothing
      return a;
    //drop starts
    var n=index.sort();
    a=a.split(c,-1);
    var b;
    i=-1;while(++i<c){
      if(i!==n[0])
        b=b?b.concat(a[i],-1):a[i];
      else
        n.shift();
    }
    if(b)return b;
    console.error("removeChannels failed. No channels left.");
  }
}


tf.layers.coord=(args)=>{return new window._guzuTF.AddCoords(args);};
tf.layers.scalar=(args)=>{return new window._guzuTF.AddScalar(args);};
tf.layers.counter=(args)=>{return new window._guzuTF.AddCounter(args);};
tf.layers.sumPooling2d=(args)=>{return new window._guzuTF.SumPooling2d(args);};
tf.layers.mutate2d=(args)=>{return new window._guzuTF.Mutation2d(args);};
tf.layers.temp=(args)=>{return new window._guzuTF.TemporaryLayer(args);};
tf.layers.bbox=(args)=>{return new window._guzuTF.BoundingBoxLayer(args);};
tf.layers.dropChannels=(args)=>{return new window._guzuTF.DropChannelsLayer(args);};
tf.layers.weight1d=(args)=>{return new window._guzuTF.Weight1DLayer(args);};
tf.layers.convWeight2d=(args)=>{return new window._guzuTF.ConvWeight2DLayer(args);};
//tf.layers.weightConv2d=(args)=>{return new window._guzuTF.WeightConv2DLayer(args);};

//some needs fixing
tf.layers.sub=window._guzuTF.guzuTfTools.sub;
tf.layers.mul=window._guzuTF.guzuTfTools.mul;
tf.layers.map=(a,b,c,d)=>window._guzuTF.guzuTfTools.map(a,b,c,d);
tf.layers.pass=window._guzuTF.guzuTfTools.pass;
