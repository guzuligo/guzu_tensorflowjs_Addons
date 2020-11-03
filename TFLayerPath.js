//Ver 0.1.0
//Helps creating multipath model
class TFLayerPath{
  
  constructor(name=""){
    this.layerPath=[];
    this.layerNames={};
    this.name=name;
  }
  
  /*
   * Add layer path
   * @param {string} index_ Path name
   * @param {tf.layers} layer_ tensorflow layer
   * @param {string | [string]} applytoIndex the index to apply this layer to. If not provided, it is considered input layer
  */
  
  add(index_,layer_,applytoIndex=-1){
    if(this.layerNames[index_]){throw("Name conflect: "+index_)}
    this.layerNames[index_]={id:this.layerPath.length,defaults:{trainable:layer_.trainable}};
    this.layerPath.push([index_,layer_,applytoIndex,-1]);//[ name, tf.layer , #to apply to, redirect ]
  }
  
  /*
   * After adding, use apply to activate tf.apply chaing to use it in a model. 
   * @param {path_} the path index that has output layer
  */
  apply(path_){
    //console.log(path_);
    if (path_===undefined) return this.apply(this.layerPath.length-1);//apply() starts
    if (typeof path_==='string')return this.apply(this.getIndex(path_));
    if (this.layerPath[path_][3]!=-1)//do redirect?
      return this.apply(this.layerPath[path_][3]);
    
    if (this.layerPath[path_][2]===-1) return this.layerPath[path_][1];//if input layer, return it
    var l=this.layerPath[path_];
    var applyto;
    //console.log("applyto")
    if (!Array.isArray(l[2]))
      applyto=this.apply(this.getIndex(l[2]));
    else {
      applyto=[];
      var i=-1;while(++i<l[2].length)
        applyto.push(this.apply(this.getIndex(l[2][i])));
    }
    //console.log("return: "+path_)
    return l[1].apply(applyto);
  }
  
  //returns tf.layer, which is specificly useful for input
  get(name){
    
    return this.getPath(name)[1];
  }
  
  //returns this.layerPath[name]
  getPath(name){
    return this.layerPath[this.getIndex(name)];
  }
  
  //returns the # index of this name
  getIndex(name){
    return (isNaN(name))? this.layerNames[name].id:name;
  }
  //{id,defaults:{trainable}}
  getInfo(name){
    if (!isNaN(name))name=this.layerPath[name][0];
    return this.layerNames[name];
  }
  /*
   * instead of using this path, use other path index
  */
  redirect(name_,to_=-1){
    this.getPath(name_)[3]=to_;
  }
  
  /*
   * @param {number | string} index_ The index of the layer or its index name
   * @param {any} layer_ if layer_ is string, replace the applyto. else, replace the tf.layer
  */
  replace(index_,layer_){
    var l_=(Array.isArray(layer_)||typeof layer_==='string')?2:1;//console.log("replacing:"+l_)
    this.getPath(index_)[l_]=layer_;
  }
  /*
   * Switch between trainable state and untrainable state
  */
  //TODO:redirect in setTrainable needs testing
  setTrainable(rootIndex_,endIndex_,targetState_,exceptIfTrainableIs_){
    //if root not defined, use the last one added
    if (rootIndex_===undefined)rootIndex_=this.layerPath.length-1;
    
    
    var r=this.getPath(rootIndex_);
    if(r[3]!==-1){//redirect?
      this.setTrainable(r[3],endIndex_,targetState_,exceptIfTrainableIs_);
      return;
    }
    
    //set value
    if(exceptIfTrainableIs_===undefined || exceptIfTrainableIs_!==this.getInfo(rootIndex_).defaults.trainable)
      r[1].trainable=targetState_;
   
    //if there is more, fix them
    
    if (rootIndex_!==endIndex_ && r[2]!==-1)
      if (Array.isArray(r[2])){
        var i=-1;while(++i<r[2].length)
          this.setTrainable(r[2][i],endIndex_,targetState_,exceptIfTrainableIs_);
      }
      else{
        this.setTrainable(r[2],endIndex_,targetState_,exceptIfTrainableIs_);}
      
  }
  
  //restores trainable
  resetTrainable(rootIndex_,endIndex_){
    
    this.setTrainable(rootIndex_,endIndex_, true,!true);
    this.setTrainable(rootIndex_,endIndex_,false,!false);
  }
  
  
}
