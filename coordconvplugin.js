//ver 1.1
class AddCoords extends tf.layers.Layer {
    
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

tf.serialization.registerClass(AddCoords);  // Needed for serialization.
//export function guzuCoordConv() {return new GuzuCoordConv();}




//to use: new AddScalar({values:[6,7]})
class AddScalar extends tf.layers.Layer {
    
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
      //console.log("res:",res);
      res[0]=it_[0].concat(tf.tensor([this.values]).tile([it_[0].shape[0],1]),-1);
      
      //if(tu++<16*16*3 && tu%15===0)res[0].print();//res[1].print();
      
      return res;
    }
}
var tu=0;
tf.serialization.registerClass(AddScalar);  // Needed for serialization.
