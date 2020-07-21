/* 
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
class AjaxTools{
    
    constructor(){
        this.xhttp=new XMLHttpRequest();
        this.data=null;
    }
    
    get(location_,onloaded_){
        this.xhttp.onloaded=function(e){this.data=this.xhttp.response;onloaded_(e);};
        this.xhttp.open("GET", location_, true);
        this.xhttp.send();
    }
} 

