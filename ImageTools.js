//Ver 0.1
class ImageTools{
    
    constructor(src,onload,whidth_height_xoffset_yoffset_AsArray,resizeWidthHeight_AsArray,renameTo_){
        //this.canvas;
        //this.context;
        //this.data;
        var me=this;
        renameTo_=renameTo_ || ("imageToCanvas_"+src);
        this.rawdata=null;
        this.alreadyExists=false;
        var canvas;
        if (document.getElementById(renameTo_)){
            canvas=document.getElementById(renameTo_);
            this.alreadyExists=true;
        }
        else
            canvas= this.canvas= document.createElement('canvas');
            
        
        var c2d=this.context=canvas.getContext("2d");
        canvas.id = renameTo_ ;
        var X=whidth_height_xoffset_yoffset_AsArray;
        var img=new Image();
        
        img.onload=function(){
           if (!X)
               X=[img.width,img.height,0,0];
           canvas.width=X[0]||img.width;
           canvas.height=X[1]||img.height;
           if (!resizeWidthHeight_AsArray)
               c2d.drawImage(img,X[2]||0,X[3]||0);
           else
               c2d.drawImage(img,X[2]||0,X[3]||0,resizeWidthHeight_AsArray[0],resizeWidthHeight_AsArray[1]);
           me.rawdata=me.context.getImageData(0,0,canvas.width,canvas.height);
            if (onload)
                onload(me);
            
            
           //hints
           //document.getElementsByTagName("body")[0].appendChild(canvas)
        };
        img.src=src;
    }
    //typeOfExeclude: 0 none, -1 ex-alpha, 1,2,3 for channel Red Green or Blue
    //returns result
    createData(typeOfExeclude=0,divideBy=1,add_=0){
        //console.log("test");
        var r=[];var i,j,k;
        var w_=this.rawdata.width;
        var h_=this.rawdata.height;
        var d_=this.rawdata.data;
        for (i=0;i<w_;i++){
            r[i]=[];
            for (j=0;j<h_;j++){
                
                k=j*w_*4+i*4;
                switch(typeOfExeclude) {

                    //execlude nothing
                    case 0:
                        r[i][j]=[
                             d_[k]/divideBy+add_
                            ,d_[k+1]/divideBy+add_
                            ,d_[k+2]/divideBy+add_
                            ,d_[k+3]/divideBy+add_
                        ];
                    break;
                    //exclude alpha
                    case -1:
                        r[i][j]=[
                            d_[k]/divideBy+add_
                            ,d_[k+1]/divideBy+add_
                            ,d_[k+2]/divideBy+add_
                        ];
                    break;
                    //choose specific channel
                    default:
                        r[i][j]=[d_[k+typeOfExeclude]/divideBy+add_];
                    break;

                }
            }
        }
        
        //return result
        return r;
    }
    
    transpose = m => m[0].map((x,i) => m.map(x => x[i]))
    
}

