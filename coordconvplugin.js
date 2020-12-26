//ver 1.3
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
      args.trainable=false;
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
      res[0]=it_[0].mul(tf.scalar(this.weight)).sub(tf.tensor(this.find)).mul(this.slope).pow(2).mul(tf.scalar(-1)).sigmoid().mul(this.scale*2);
      
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
         console.log("ee")
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


tf.layers.coord=(args)=>{return new window._guzuTF.AddCoords(args);};
tf.layers.scalar=(args)=>{return new window._guzuTF.AddScalar(args);};
tf.layers.counter=(args)=>{return new window._guzuTF.AddCounter(args);};
tf.layers.sumPooling2d=(args)=>{return new window._guzuTF.SumPooling2d(args);};
