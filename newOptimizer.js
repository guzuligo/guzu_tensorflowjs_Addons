class optX{
    constructor(){
        this.mutations=[];
    }


    getMutation(model,seed){
        var w=model.getWeights();
        
        if(!seed)seed=Math.random()*1000;
        if(!isNaN(seed))seed=[0,1,seed];//make seed array
        
        
        for (var i in w){
           w[i]=w[i].add(tf.randomNormal(w[i].shape,seed[0],seed[1],'float32',seed[2]))
        }

        var m={
            seed:seed,
            loss:undefined,
        };
        
        this.mutations.push(m);
        return w;
    }

    setMutation(model,seed){
        model.setWeights(model.getMutation(model,seed));
    }



}