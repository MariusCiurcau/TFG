(function(){'use strict';var n;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba=typeof Object.defineProperties=="function"?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var fa=ca(this);function w(a,b){if(b)a:{var c=fa;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&b!=null&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
w("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(Math.random()*1E9>>>0)+"_",e=0;return b});
w("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=fa[b[c]];typeof d==="function"&&typeof d.prototype[a]!="function"&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ia(aa(this))}})}return a});
function ia(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
function ja(a){return a.raw=a}
function ka(a,b){a.raw=b;return a}
function x(a){var b=typeof Symbol!="undefined"&&Symbol.iterator&&a[Symbol.iterator];if(b)return b.call(a);if(typeof a.length=="number")return{next:aa(a)};throw Error(String(a)+" is not an iterable or ArrayLike");}
function la(a){if(!(a instanceof Array)){a=x(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function ma(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var oa=typeof Object.assign=="function"?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)ma(d,e)&&(a[e]=d[e])}return a};
w("Object.assign",function(a){return a||oa});
var pa=typeof Object.create=="function"?Object.create:function(a){function b(){}
b.prototype=a;return new b},qa=function(){function a(){function c(){}
new c;Reflect.construct(c,[],function(){});
return new c instanceof c}
if(typeof Reflect!="undefined"&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(c,d,e){c=b(c,d);e&&Reflect.setPrototypeOf(c,e.prototype);return c}}return function(c,d,e){e===void 0&&(e=c);
e=pa(e.prototype||Object.prototype);return Function.prototype.apply.call(c,e,d)||e}}(),ra;
if(typeof Object.setPrototypeOf=="function")ra=Object.setPrototypeOf;else{var sa;a:{var ta={a:!0},ua={};try{ua.__proto__=ta;sa=ua.a;break a}catch(a){}sa=!1}ra=sa?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var wa=ra;
function y(a,b){a.prototype=pa(b.prototype);a.prototype.constructor=a;if(wa)wa(a,b);else for(var c in b)if(c!="prototype")if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.Ba=b.prototype}
function xa(){this.A=!1;this.m=null;this.i=void 0;this.h=1;this.v=this.l=0;this.P=this.j=null}
function ya(a){if(a.A)throw new TypeError("Generator is already running");a.A=!0}
xa.prototype.H=function(a){this.i=a};
function za(a,b){a.j={exception:b,jd:!0};a.h=a.l||a.v}
xa.prototype.return=function(a){this.j={return:a};this.h=this.v};
xa.prototype.yield=function(a,b){this.h=b;return{value:a}};
xa.prototype.B=function(a){this.h=a};
function Aa(a,b,c){a.l=b;c!=void 0&&(a.v=c)}
function Ba(a){a.l=0;var b=a.j.exception;a.j=null;return b}
function Ca(a){var b=a.P.splice(0)[0];(b=a.j=a.j||b)?b.jd?a.h=a.l||a.v:b.B!=void 0&&a.v<b.B?(a.h=b.B,a.j=null):a.h=a.v:a.h=0}
function Da(a){this.h=new xa;this.i=a}
function Ea(a,b){ya(a.h);var c=a.h.m;if(c)return Fa(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return Ga(a)}
function Fa(a,b,c,d){try{var e=b.call(a.h.m,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.A=!1,e;var f=e.value}catch(g){return a.h.m=null,za(a.h,g),Ga(a)}a.h.m=null;d.call(a.h,f);return Ga(a)}
function Ga(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.A=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,za(a.h,c)}a.h.A=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.jd)throw b.exception;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Ha(a){this.next=function(b){ya(a.h);a.h.m?b=Fa(a,a.h.m.next,b,a.h.H):(a.h.H(b),b=Ga(a));return b};
this.throw=function(b){ya(a.h);a.h.m?b=Fa(a,a.h.m["throw"],b,a.h.H):(za(a.h,b),b=Ga(a));return b};
this.return=function(b){return Ea(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ia(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function A(a){return Ia(new Ha(new Da(a)))}
function B(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
w("Reflect",function(a){return a?a:{}});
w("Reflect.construct",function(){return qa});
w("Reflect.setPrototypeOf",function(a){return a?a:wa?function(b,c){try{return wa(b,c),!0}catch(d){return!1}}:null});
w("Promise",function(a){function b(g){this.h=0;this.j=void 0;this.i=[];this.A=!1;var h=this.l();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(this.h==null){this.h=[];var h=this;this.j(function(){h.v()})}this.h.push(g)};
var e=fa.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.v=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(l){this.l(l)}}}this.h=null};
c.prototype.l=function(g){this.j(function(){throw g;})};
b.prototype.l=function(){function g(l){return function(m){k||(k=!0,l.call(h,m))}}
var h=this,k=!1;return{resolve:g(this.da),reject:g(this.v)}};
b.prototype.da=function(g){if(g===this)this.v(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.ia(g);else{a:switch(typeof g){case "object":var h=g!=null;break a;case "function":h=!0;break a;default:h=!1}h?this.ba(g):this.m(g)}};
b.prototype.ba=function(g){var h=void 0;try{h=g.then}catch(k){this.v(k);return}typeof h=="function"?this.xa(h,g):this.m(g)};
b.prototype.v=function(g){this.H(2,g)};
b.prototype.m=function(g){this.H(1,g)};
b.prototype.H=function(g,h){if(this.h!=0)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.h);this.h=g;this.j=h;this.h===2&&this.ga();this.P()};
b.prototype.ga=function(){var g=this;e(function(){if(g.W()){var h=fa.console;typeof h!=="undefined"&&h.error(g.j)}},1)};
b.prototype.W=function(){if(this.A)return!1;var g=fa.CustomEvent,h=fa.Event,k=fa.dispatchEvent;if(typeof k==="undefined")return!0;typeof g==="function"?g=new g("unhandledrejection",{cancelable:!0}):typeof h==="function"?g=new h("unhandledrejection",{cancelable:!0}):(g=fa.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.j;return k(g)};
b.prototype.P=function(){if(this.i!=null){for(var g=0;g<this.i.length;++g)f.i(this.i[g]);this.i=null}};
var f=new c;b.prototype.ia=function(g){var h=this.l();g.Zb(h.resolve,h.reject)};
b.prototype.xa=function(g,h){var k=this.l();try{g.call(h,k.resolve,k.reject)}catch(l){k.reject(l)}};
b.prototype.then=function(g,h){function k(t,r){return typeof t=="function"?function(v){try{l(t(v))}catch(u){m(u)}}:r}
var l,m,p=new b(function(t,r){l=t;m=r});
this.Zb(k(g,l),k(h,m));return p};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.Zb=function(g,h){function k(){switch(l.h){case 1:g(l.j);break;case 2:h(l.j);break;default:throw Error("Unexpected state: "+l.h);}}
var l=this;this.i==null?f.i(k):this.i.push(k);this.A=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var l=x(g),m=l.next();!m.done;m=l.next())d(m.value).Zb(h,k)})};
b.all=function(g){var h=x(g),k=h.next();return k.done?d([]):new b(function(l,m){function p(v){return function(u){t[v]=u;r--;r==0&&l(t)}}
var t=[],r=0;do t.push(void 0),r++,d(k.value).Zb(p(t.length-1),m),k=h.next();while(!k.done)})};
return b});
w("Object.setPrototypeOf",function(a){return a||wa});
w("Symbol.dispose",function(a){return a?a:Symbol("Symbol.dispose")});
w("WeakMap",function(a){function b(k){this.h=(h+=Math.random()+1).toString();if(k){k=x(k);for(var l;!(l=k.next()).done;)l=l.value,this.set(l[0],l[1])}}
function c(){}
function d(k){var l=typeof k;return l==="object"&&k!==null||l==="function"}
function e(k){if(!ma(k,g)){var l=new c;ba(k,g,{value:l})}}
function f(k){var l=Object[k];l&&(Object[k]=function(m){if(m instanceof c)return m;Object.isExtensible(m)&&e(m);return l(m)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),l=Object.seal({}),m=new a([[k,2],[l,3]]);if(m.get(k)!=2||m.get(l)!=3)return!1;m.delete(k);m.set(l,4);return!m.has(k)&&m.get(l)==4}catch(p){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,l){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!ma(k,g))throw Error("WeakMap key fail: "+k);k[g][this.h]=l;return this};
b.prototype.get=function(k){return d(k)&&ma(k,g)?k[g][this.h]:void 0};
b.prototype.has=function(k){return d(k)&&ma(k,g)&&ma(k[g],this.h)};
b.prototype.delete=function(k){return d(k)&&ma(k,g)&&ma(k[g],this.h)?delete k[g][this.h]:!1};
return b});
w("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var l=h[1];return ia(function(){if(l){for(;l.head!=h[1];)l=l.previous;for(;l.next!=l.head;)return l=l.next,{done:!1,value:k(l)};l=null}return{done:!0,value:void 0}})}
function d(h,k){var l=k&&typeof k;l=="object"||l=="function"?f.has(k)?l=f.get(k):(l=""+ ++g,f.set(k,l)):l="p_"+k;var m=h[0][l];if(m&&ma(h[0],l))for(h=0;h<m.length;h++){var p=m[h];if(k!==k&&p.key!==p.key||k===p.key)return{id:l,list:m,index:h,entry:p}}return{id:l,list:m,index:-1,entry:void 0}}
function e(h){this[0]={};this[1]=b();this.size=0;if(h){h=x(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var h=Object.seal({x:4}),k=new a(x([[h,"s"]]));if(k.get(h)!="s"||k.size!=1||k.get({x:4})||k.set({x:4},"t")!=k||k.size!=2)return!1;var l=k.entries(),m=l.next();if(m.done||m.value[0]!=h||m.value[1]!="s")return!1;m=l.next();return m.done||m.value[0].x!=4||m.value[1]!="t"||!l.next().done?!1:!0}catch(p){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=h===0?0:h;var l=d(this,h);l.list||(l.list=this[0][l.id]=[]);l.entry?l.entry.value=k:(l.entry={next:this[1],previous:this[1].previous,head:this[1],key:h,value:k},l.list.push(l.entry),this[1].previous.next=l.entry,this[1].previous=l.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this[0][h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this[0]={};this[1]=this[1].previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var l=this.entries(),m;!(m=l.next()).done;)m=m.value,h.call(k,m[1],m[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
function Ja(a,b,c){if(a==null)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
w("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ja(this,b,"endsWith");b+="";c===void 0&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;e>0&&c>0;)if(d[--c]!=b[--e])return!1;return e<=0}});
function Ma(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
w("Array.prototype.entries",function(a){return a?a:function(){return Ma(this,function(b,c){return[b,c]})}});
w("Array.prototype.keys",function(a){return a?a:function(){return Ma(this,function(b){return b})}});
w("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ja(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
w("Number.isFinite",function(a){return a?a:function(b){return typeof b!=="number"?!1:!isNaN(b)&&b!==Infinity&&b!==-Infinity}});
w("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
w("Set",function(a){function b(c){this.h=new Map;if(c){c=x(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var c=Object.seal({x:4}),d=new a(x([c]));if(!d.has(c)||d.size!=1||d.add(c)!=d||d.size!=1||d.add({x:4})!=d||d.size!=2)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||f.value[0].x!=4||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=c===0?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
w("Array.prototype.values",function(a){return a?a:function(){return Ma(this,function(b,c){return c})}});
w("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
w("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
w("Number.isSafeInteger",function(a){return a?a:function(b){return Number.isInteger(b)&&Math.abs(b)<=Number.MAX_SAFE_INTEGER}});
w("Math.trunc",function(a){return a?a:function(b){b=Number(b);if(isNaN(b)||b===Infinity||b===-Infinity||b===0)return b;var c=Math.floor(Math.abs(b));return b<0?-c:c}});
w("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)ma(b,d)&&c.push(b[d]);return c}});
w("Object.is",function(a){return a?a:function(b,c){return b===c?b!==0||1/b===1/c:b!==b&&c!==c}});
w("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(c<0&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
w("String.prototype.includes",function(a){return a?a:function(b,c){return Ja(this,b,"includes").indexOf(b,c||0)!==-1}});
w("Number.isNaN",function(a){return a?a:function(b){return typeof b==="number"&&isNaN(b)}});
w("Array.from",function(a){return a?a:function(b,c,d){c=c!=null?c:function(h){return h};
var e=[],f=typeof Symbol!="undefined"&&Symbol.iterator&&b[Symbol.iterator];if(typeof f=="function"){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
w("Math.clz32",function(a){return a?a:function(b){b=Number(b)>>>0;if(b===0)return 32;var c=0;(b&4294901760)===0&&(b<<=16,c+=16);(b&4278190080)===0&&(b<<=8,c+=8);(b&4026531840)===0&&(b<<=4,c+=4);(b&3221225472)===0&&(b<<=2,c+=2);(b&2147483648)===0&&c++;return c}});
w("Math.log10",function(a){return a?a:function(b){return Math.log(b)/Math.LN10}});
w("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)ma(b,d)&&c.push([d,b[d]]);return c}});
w("globalThis",function(a){return a||fa});/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var Na=Na||{},C=this||self;function D(a,b,c){a=a.split(".");c=c||C;a[0]in c||typeof c.execScript=="undefined"||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||b===void 0?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function Oa(a,b){var c=E("CLOSURE_FLAGS");a=c&&c[a];return a!=null?a:b}
function E(a,b){a=a.split(".");b=b||C;for(var c=0;c<a.length;c++)if(b=b[a[c]],b==null)return null;return b}
function Pa(a){var b=typeof a;return b!="object"?b:a?Array.isArray(a)?"array":b:"null"}
function Qa(a){var b=Pa(a);return b=="array"||b=="object"&&typeof a.length=="number"}
function Ra(a){var b=typeof a;return b=="object"&&a!=null||b=="function"}
function Sa(a){return Object.prototype.hasOwnProperty.call(a,Ta)&&a[Ta]||(a[Ta]=++Ua)}
var Ta="closure_uid_"+(Math.random()*1E9>>>0),Ua=0;function Va(a,b,c){return a.call.apply(a.bind,arguments)}
function Wa(a,b,c){if(!a)throw Error();if(arguments.length>2){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Xa(a,b,c){Xa=Function.prototype.bind&&Function.prototype.bind.toString().indexOf("native code")!=-1?Va:Wa;return Xa.apply(null,arguments)}
function Ya(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}}
function Za(){return Date.now()}
function $a(a,b){function c(){}
c.prototype=b.prototype;a.Ba=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.base=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
function ab(a){return a}
;function bb(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,bb);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));b!==void 0&&(this.cause=b)}
$a(bb,Error);bb.prototype.name="CustomError";function cb(a){a=a.url;var b=/[?&]dsh=1(&|$)/.test(a);this.j=!b&&/[?&]ae=1(&|$)/.test(a);this.l=!b&&/[?&]ae=2(&|$)/.test(a);if((this.h=/[?&]adurl=([^&]*)/.exec(a))&&this.h[1]){try{var c=decodeURIComponent(this.h[1])}catch(d){c=null}this.i=c}}
;var db=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};var eb;function fb(){if(eb===void 0){var a=null,b=C.trustedTypes;if(b&&b.createPolicy){try{a=b.createPolicy("goog#html",{createHTML:ab,createScript:ab,createScriptURL:ab})}catch(c){C.console&&C.console.error(c.message)}eb=a}else eb=a}return eb}
;function gb(a,b){this.h=a===hb&&b||""}
gb.prototype.toString=function(){return this.h};
function ib(a){return new gb(hb,a)}
var hb={};ib("");function jb(a){this.h=a}
jb.prototype.toString=function(){return this.h+""};
function kb(a){if(a instanceof jb&&a.constructor===jb)return a.h;Pa(a);return"type_error:TrustedResourceUrl"}
var lb={};function mb(a){var b=fb();a=b?b.createScriptURL(a):a;return new jb(a,lb)}
;/*

 SPDX-License-Identifier: Apache-2.0
*/
var nb=ja([""]),ob=ka(["\x00"],["\\0"]),pb=ka(["\n"],["\\n"]),qb=ka(["\x00"],["\\u0000"]);function rb(a){return a.toString().indexOf("`")===-1}
rb(function(a){return a(nb)})||rb(function(a){return a(ob)})||rb(function(a){return a(pb)})||rb(function(a){return a(qb)});function sb(a){this.h=a}
sb.prototype.toString=function(){return this.h};
var tb=new sb("about:invalid#zClosurez");function ub(a){this.pe=a}
function vb(a){return new ub(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var wb=[vb("data"),vb("http"),vb("https"),vb("mailto"),vb("ftp"),new ub(function(a){return/^[^:]*([/?#]|$)/.test(a)})],xb=/^\s*(?!javascript:)(?:[\w+.-]+:|[^:/?#]*(?:[/?#]|$))/i;
function yb(a){if(a instanceof sb)if(a instanceof sb)a=a.h;else throw Error("");else a=xb.test(a)?a:void 0;return a}
;function zb(a,b){b=yb(b);b!==void 0&&(a.href=b)}
;function Ab(){this.h=Bb[0].toLowerCase()}
Ab.prototype.toString=function(){return this.h};var Cb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if(typeof a==="string")return typeof b!=="string"||b.length!=1?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},Db=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=typeof a==="string"?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},Eb=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f=typeof a==="string"?a.split(""):a,g=0;g<c;g++)if(g in f){var h=f[g];
b.call(void 0,h,g,a)&&(d[e++]=h)}return d},Fb=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e=typeof a==="string"?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},Gb=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
Db(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function Hb(a,b){a:{for(var c=a.length,d=typeof a==="string"?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return b<0?null:typeof a==="string"?a.charAt(b):a[b]}
function Ib(a,b){b=Cb(a,b);var c;(c=b>=0)&&Array.prototype.splice.call(a,b,1);return c}
function Jb(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Qa(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function Kb(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function Lb(a){var b=Mb,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function Nb(a){for(var b in a)return!1;return!0}
function Ob(a,b){if(a!==null&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function Pb(a){return a!==null&&"privembed"in a?a.privembed:!1}
function Qb(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function Rb(a){var b={},c;for(c in a)b[c]=a[c];return b}
function Sb(a){if(!a||typeof a!=="object")return a;if(typeof a.clone==="function")return a.clone();if(typeof Map!=="undefined"&&a instanceof Map)return new Map(a);if(typeof Set!=="undefined"&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:typeof ArrayBuffer!=="function"||typeof ArrayBuffer.isView!=="function"||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=Sb(a[c]);return b}
var Tb="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function Ub(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<Tb.length;f++)c=Tb[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function Vb(a){this.h=a}
Vb.prototype.toString=function(){return this.h.toString()};function Wb(a){var b="true".toString(),c=[new Ab];if(c.length===0)throw Error("");if(c.map(function(d){if(d instanceof Ab)d=d.h;else throw Error("");return d}).every(function(d){return"data-loaded".indexOf(d)!==0}))throw Error('Attribute "data-loaded" does not match any of the allowed prefixes.');
a.setAttribute("data-loaded",b)}
;function Xb(a,b){throw Error(b===void 0?"unexpected value "+a+"!":b);}
;var Yb="alternate author bookmark canonical cite help icon license modulepreload next prefetch dns-prefetch prerender preconnect preload prev search subresource".split(" ");function Zb(a,b){if(b instanceof jb)a.href=kb(b).toString();else{if(Yb.indexOf("stylesheet")===-1)throw Error('TrustedResourceUrl href attribute required with rel="stylesheet"');b=yb(b);if(b===void 0)return;a.href=b}a.rel="stylesheet"}
;function $b(a){var b,c;return(a=(c=(b=a.document).querySelector)==null?void 0:c.call(b,"script[nonce]"))?a.nonce||a.getAttribute("nonce")||"":""}
;function ac(a){this.h=a}
ac.prototype.toString=function(){return this.h.toString()};function bc(a){var b=$b(a.ownerDocument&&a.ownerDocument.defaultView||window);b&&a.setAttribute("nonce",b)}
function cc(a,b){if(b instanceof ac)b=b.h;else throw Error("");a.textContent=b;bc(a)}
function dc(a,b){a.src=kb(b);bc(a)}
;function ec(a,b){a.__closure__error__context__984382||(a.__closure__error__context__984382={});a.__closure__error__context__984382.severity=b}
;function fc(a){var b=E("window.location.href");a==null&&(a='Unknown Error of type "null/undefined"');if(typeof a==="string")return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||C.$googDebugFname||b}catch(g){e="Not available",c=!0}b=hc(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(c==
null){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,ic[c])c=ic[c];else{c=String(c);if(!ic[c]){var f=/function\s+([^\(]+)/m.exec(c);ic[c]=f?f[1]:"[Anonymous]"}c=ic[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";typeof a.toString==="function"&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}return{message:a.message,
name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:b}}
function hc(a,b){b||(b={});b[jc(a)]=!0;var c=a.stack||"";(a=a.cause)&&!b[jc(a)]&&(c+="\nCaused by: ",a.stack&&a.stack.indexOf(a.toString())==0||(c+=typeof a==="string"?a:a.message+"\n"),c+=hc(a,b));return c}
function jc(a){var b="";typeof a.toString==="function"&&(b=""+a);return b+a.stack}
var ic={};function kc(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var lc=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function mc(a){return a?decodeURI(a):a}
function nc(a,b){return b.match(lc)[a]||null}
function oc(a){return mc(nc(3,a))}
function pc(a){var b=a.match(lc);a=b[5];var c=b[6];b=b[7];var d="";a&&(d+=a);c&&(d+="?"+c);b&&(d+="#"+b);return d}
function qc(a){var b=a.indexOf("#");return b<0?a:a.slice(0,b)}
function rc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)rc(a,String(b[d]),c);else b!=null&&c.push(a+(b===""?"":"="+encodeURIComponent(String(b))))}
function sc(a){var b=[],c;for(c in a)rc(c,a[c],b);return b.join("&")}
function tc(a,b){b=sc(b);if(b){var c=a.indexOf("#");c<0&&(c=a.length);var d=a.indexOf("?");if(d<0||d>c){d=c;var e=""}else e=a.substring(d+1,c);a=[a.slice(0,d),e,a.slice(c)];c=a[1];a[1]=b?c?c+"&"+b:b:c;b=a[0]+(a[1]?"?"+a[1]:"")+a[2]}else b=a;return b}
function uc(a,b,c,d){for(var e=c.length;(b=a.indexOf(c,b))>=0&&b<d;){var f=a.charCodeAt(b-1);if(f==38||f==63)if(f=a.charCodeAt(b+e),!f||f==61||f==38||f==35)return b;b+=e+1}return-1}
var vc=/#|$/,wc=/[?&]($|#)/;function xc(a,b){for(var c=a.search(vc),d=0,e,f=[];(e=uc(a,d,b,c))>=0;)f.push(a.substring(d,e)),d=Math.min(a.indexOf("&",e)+1||c,c);f.push(a.slice(d));return f.join("").replace(wc,"$1")}
;function yc(a){this.h=a}
;function zc(a,b,c){this.l=a;this.j=b;this.fields=c||[];this.h=new Map}
n=zc.prototype;n.Kd=function(a){var b=B.apply(1,arguments),c=this.yc(b);c?c.push(new yc(a)):this.wd(a,b)};
n.wd=function(a){var b=this.Sc(B.apply(1,arguments));this.h.set(b,[new yc(a)])};
n.yc=function(){var a=this.Sc(B.apply(0,arguments));return this.h.has(a)?this.h.get(a):void 0};
n.ce=function(){var a=this.yc(B.apply(0,arguments));return a&&a.length?a[0]:void 0};
n.clear=function(){this.h.clear()};
n.Sc=function(){var a=B.apply(0,arguments);return a?a.join(","):"key"};function Ac(a,b){zc.call(this,a,3,b)}
y(Ac,zc);Ac.prototype.i=function(a){var b=B.apply(1,arguments),c=0,d=this.ce(b);d&&(c=d.h);this.wd(c+a,b)};function Bc(a,b){zc.call(this,a,2,b)}
y(Bc,zc);Bc.prototype.record=function(a){this.Kd(a,B.apply(1,arguments))};function Cc(a){a&&typeof a.dispose=="function"&&a.dispose()}
;function Dc(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];Qa(d)?Dc.apply(null,d):Cc(d)}}
;function G(){this.V=this.V;this.v=this.v}
G.prototype.V=!1;G.prototype.dispose=function(){this.V||(this.V=!0,this.U())};
function Ec(a,b){a.addOnDisposeCallback(Ya(Cc,b))}
G.prototype.addOnDisposeCallback=function(a,b){this.V?b!==void 0?a.call(b):a():(this.v||(this.v=[]),this.v.push(b!==void 0?Xa(a,b):a))};
G.prototype.U=function(){if(this.v)for(;this.v.length;)this.v.shift()()};function Fc(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
Fc.prototype.stopPropagation=function(){this.j=!0};
Fc.prototype.preventDefault=function(){this.defaultPrevented=!0};var Gc=function(){if(!C.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{var c=function(){};
C.addEventListener("test",c,b);C.removeEventListener("test",c,b)}catch(d){}return a}();var Hc=Oa(610401301,!1),Ic=Oa(188588736,Oa(1,!0));function Jc(){var a=C.navigator;return a&&(a=a.userAgent)?a:""}
var Kc,Lc=C.navigator;Kc=Lc?Lc.userAgentData||null:null;function Mc(a){return Hc?Kc?Kc.brands.some(function(b){return(b=b.brand)&&b.indexOf(a)!=-1}):!1:!1}
function H(a){return Jc().indexOf(a)!=-1}
;function Nc(){return Hc?!!Kc&&Kc.brands.length>0:!1}
function Oc(){return Nc()?!1:H("Opera")}
function Pc(){return H("Firefox")||H("FxiOS")}
function Qc(){return Nc()?Mc("Chromium"):(H("Chrome")||H("CriOS"))&&!(Nc()?0:H("Edge"))||H("Silk")}
;function Rc(){return Hc?!!Kc&&!!Kc.platform:!1}
function Sc(){return H("iPhone")&&!H("iPod")&&!H("iPad")}
;function Tc(a){Tc[" "](a);return a}
Tc[" "]=function(){};var Uc=Oc(),Vc=Nc()?!1:H("Trident")||H("MSIE"),Wc=H("Edge"),Xc=H("Gecko")&&!(Jc().toLowerCase().indexOf("webkit")!=-1&&!H("Edge"))&&!(H("Trident")||H("MSIE"))&&!H("Edge"),Yc=Jc().toLowerCase().indexOf("webkit")!=-1&&!H("Edge");Yc&&H("Mobile");Rc()||H("Macintosh");Rc()||H("Windows");(Rc()?Kc.platform==="Linux":H("Linux"))||Rc()||H("CrOS");var Zc=Rc()?Kc.platform==="Android":H("Android");Sc();H("iPad");H("iPod");Sc()||H("iPad")||H("iPod");Jc().toLowerCase().indexOf("kaios");function $c(a,b){Fc.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
$a($c,Fc);var ad={2:"touch",3:"pen",4:"mouse"};
$c.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;if(b=a.relatedTarget){if(Xc){a:{try{Tc(b.nodeName);var e=!0;break a}catch(f){}e=!1}e||(b=null)}}else c=="mouseover"?b=a.fromElement:c=="mouseout"&&(b=a.toElement);this.relatedTarget=b;d?(this.clientX=d.clientX!==void 0?d.clientX:d.pageX,this.clientY=d.clientY!==void 0?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||
0):(this.clientX=a.clientX!==void 0?a.clientX:a.pageX,this.clientY=a.clientY!==void 0?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||(c=="keypress"?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType=typeof a.pointerType==="string"?a.pointerType:ad[a.pointerType]||"";this.state=a.state;
this.i=a;a.defaultPrevented&&$c.Ba.preventDefault.call(this)};
$c.prototype.stopPropagation=function(){$c.Ba.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
$c.prototype.preventDefault=function(){$c.Ba.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var bd="closure_listenable_"+(Math.random()*1E6|0);var cd=0;function dd(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.fc=e;this.key=++cd;this.Pb=this.Yb=!1}
function ed(a){a.Pb=!0;a.listener=null;a.proxy=null;a.src=null;a.fc=null}
;function fd(a){this.src=a;this.listeners={};this.h=0}
fd.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=gd(a,b,d,e);g>-1?(b=a[g],c||(b.Yb=!1)):(b=new dd(b,this.src,f,!!d,e),b.Yb=c,a.push(b));return b};
fd.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=gd(e,b,c,d);return b>-1?(ed(e[b]),Array.prototype.splice.call(e,b,1),e.length==0&&(delete this.listeners[a],this.h--),!0):!1};
function hd(a,b){var c=b.type;c in a.listeners&&Ib(a.listeners[c],b)&&(ed(b),a.listeners[c].length==0&&(delete a.listeners[c],a.h--))}
function gd(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.Pb&&f.listener==b&&f.capture==!!c&&f.fc==d)return e}return-1}
;var id="closure_lm_"+(Math.random()*1E6|0),jd={},kd=0;function ld(a,b,c,d,e){if(d&&d.once)md(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)ld(a,b[f],c,d,e);else c=nd(c),a&&a[bd]?a.listen(b,c,Ra(d)?!!d.capture:!!d,e):od(a,b,c,!1,d,e)}
function od(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Ra(e)?!!e.capture:!!e,h=pd(a);h||(a[id]=h=new fd(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=qd();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)Gc||(e=g),e===void 0&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(rd(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");kd++}}
function qd(){function a(c){return b.call(a.src,a.listener,c)}
var b=sd;return a}
function md(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)md(a,b[f],c,d,e);else c=nd(c),a&&a[bd]?a.h.add(String(b),c,!0,Ra(d)?!!d.capture:!!d,e):od(a,b,c,!0,d,e)}
function td(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)td(a,b[f],c,d,e);else(d=Ra(d)?!!d.capture:!!d,c=nd(c),a&&a[bd])?a.h.remove(String(b),c,d,e):a&&(a=pd(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=gd(b,c,d,e)),(c=a>-1?b[a]:null)&&ud(c))}
function ud(a){if(typeof a!=="number"&&a&&!a.Pb){var b=a.src;if(b&&b[bd])hd(b.h,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(rd(c),d):b.addListener&&b.removeListener&&b.removeListener(d);kd--;(c=pd(b))?(hd(c,a),c.h==0&&(c.src=null,b[id]=null)):ed(a)}}}
function rd(a){return a in jd?jd[a]:jd[a]="on"+a}
function sd(a,b){if(a.Pb)a=!0;else{b=new $c(b,this);var c=a.listener,d=a.fc||a.src;a.Yb&&ud(a);a=c.call(d,b)}return a}
function pd(a){a=a[id];return a instanceof fd?a:null}
var vd="__closure_events_fn_"+(Math.random()*1E9>>>0);function nd(a){if(typeof a==="function")return a;a[vd]||(a[vd]=function(b){return a.handleEvent(b)});
return a[vd]}
;function wd(){G.call(this);this.h=new fd(this);this.Za=this;this.ga=null}
$a(wd,G);wd.prototype[bd]=!0;n=wd.prototype;n.addEventListener=function(a,b,c,d){ld(this,a,b,c,d)};
n.removeEventListener=function(a,b,c,d){td(this,a,b,c,d)};
function xd(a,b){var c=a.ga;if(c){var d=[];for(var e=1;c;c=c.ga)d.push(c),++e}a=a.Za;c=b.type||b;typeof b==="string"?b=new Fc(b,a):b instanceof Fc?b.target=b.target||a:(e=b,b=new Fc(c,a),Ub(b,e));e=!0;if(d)for(var f=d.length-1;!b.j&&f>=0;f--){var g=b.h=d[f];e=yd(g,c,!0,b)&&e}b.j||(g=b.h=a,e=yd(g,c,!0,b)&&e,b.j||(e=yd(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=yd(g,c,!1,b)&&e}
n.U=function(){wd.Ba.U.call(this);this.removeAllListeners();this.ga=null};
n.listen=function(a,b,c,d){return this.h.add(String(a),b,!1,c,d)};
n.removeAllListeners=function(a){if(this.h){var b=this.h;a=a&&a.toString();var c=0,d;for(d in b.listeners)if(!a||d==a){for(var e=b.listeners[d],f=0;f<e.length;f++)++c,ed(e[f]);delete b.listeners[d];b.h--}b=c}else b=0;return b};
function yd(a,b,c,d){b=a.h.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.Pb&&g.capture==c){var h=g.listener,k=g.fc||g.src;g.Yb&&hd(a.h,g);e=h.call(k,d)!==!1&&e}}return e&&!d.defaultPrevented}
;function zd(a,b){this.j=a;this.l=b;this.i=0;this.h=null}
zd.prototype.get=function(){if(this.i>0){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function Ad(a,b){a.l(b);a.i<100&&(a.i++,b.next=a.h,a.h=b)}
;function Bd(){}
function Cd(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;"ARTICLE SECTION NAV ASIDE H1 H2 H3 H4 H5 H6 HEADER FOOTER ADDRESS P HR PRE BLOCKQUOTE OL UL LH LI DL DT DD FIGURE FIGCAPTION MAIN DIV EM STRONG SMALL S CITE Q DFN ABBR RUBY RB RT RTC RP DATA TIME CODE VAR SAMP KBD SUB SUP I B U MARK BDI BDO SPAN BR WBR INS DEL PICTURE PARAM TRACK MAP TABLE CAPTION COLGROUP COL TBODY THEAD TFOOT TR TD TH SELECT DATALIST OPTGROUP OPTION OUTPUT PROGRESS METER FIELDSET LEGEND DETAILS SUMMARY MENU DIALOG SLOT CANVAS FONT CENTER ACRONYM BASEFONT BIG DIR HGROUP STRIKE TT".split(" ").concat(["BUTTON",
"INPUT"]);function Dd(a,b){this.x=a!==void 0?a:0;this.y=b!==void 0?b:0}
n=Dd.prototype;n.clone=function(){return new Dd(this.x,this.y)};
n.equals=function(a){return a instanceof Dd&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
n.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
n.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
n.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};
n.scale=function(a,b){this.x*=a;this.y*=typeof b==="number"?b:a;return this};function Ed(a,b){this.width=a;this.height=b}
n=Ed.prototype;n.clone=function(){return new Ed(this.width,this.height)};
n.aspectRatio=function(){return this.width/this.height};
n.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
n.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
n.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};
n.scale=function(a,b){this.width*=a;this.height*=typeof b==="number"?b:a;return this};function Fd(a){var b=document;return typeof a==="string"?b.getElementById(a):a}
function Gd(a){var b=document;a=String(a);b.contentType==="application/xhtml+xml"&&(a=a.toLowerCase());return b.createElement(a)}
function Hd(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;var Id;function Jd(){var a=C.MessageChannel;typeof a==="undefined"&&typeof window!=="undefined"&&window.postMessage&&window.addEventListener&&!H("Presto")&&(a=function(){var e=Gd("IFRAME");e.style.display="none";document.documentElement.appendChild(e);var f=e.contentWindow;e=f.document;e.open();e.close();var g="callImmediate"+Math.random(),h=f.location.protocol=="file:"?"*":f.location.protocol+"//"+f.location.host;e=Xa(function(k){if((h=="*"||k.origin==h)&&k.data==g)this.port1.onmessage()},this);
f.addEventListener("message",e,!1);this.port1={};this.port2={postMessage:function(){f.postMessage(g,h)}}});
if(typeof a!=="undefined"){var b=new a,c={},d=c;b.port1.onmessage=function(){if(c.next!==void 0){c=c.next;var e=c.Wc;c.Wc=null;e()}};
return function(e){d.next={Wc:e};d=d.next;b.port2.postMessage(0)}}return function(e){C.setTimeout(e,0)}}
;function Kd(a){C.setTimeout(function(){throw a;},0)}
;function Ld(){this.i=this.h=null}
Ld.prototype.add=function(a,b){var c=Md.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
Ld.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var Md=new zd(function(){return new Nd},function(a){return a.reset()});
function Nd(){this.next=this.scope=this.h=null}
Nd.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
Nd.prototype.reset=function(){this.next=this.scope=this.h=null};var Od,Pd=!1,Qd=new Ld;function Rd(a,b){Od||Sd();Pd||(Od(),Pd=!0);Qd.add(a,b)}
function Sd(){if(C.Promise&&C.Promise.resolve){var a=C.Promise.resolve(void 0);Od=function(){a.then(Td)}}else Od=function(){var b=Td;
typeof C.setImmediate!=="function"||C.Window&&C.Window.prototype&&C.Window.prototype.setImmediate==C.setImmediate?(Id||(Id=Jd()),Id(b)):C.setImmediate(b)}}
function Td(){for(var a;a=Qd.remove();){try{a.h.call(a.scope)}catch(b){Kd(b)}Ad(Md,a)}Pd=!1}
;function Ud(a){this.h=0;this.A=void 0;this.l=this.i=this.j=null;this.v=this.m=!1;if(a!=Bd)try{var b=this;a.call(void 0,function(c){Vd(b,2,c)},function(c){Vd(b,3,c)})}catch(c){Vd(this,3,c)}}
function Wd(){this.next=this.context=this.h=this.i=this.child=null;this.j=!1}
Wd.prototype.reset=function(){this.context=this.h=this.i=this.child=null;this.j=!1};
var Xd=new zd(function(){return new Wd},function(a){a.reset()});
function Yd(a,b,c){var d=Xd.get();d.i=a;d.h=b;d.context=c;return d}
function Zd(a){return new Ud(function(b,c){c(a)})}
Ud.prototype.then=function(a,b,c){return $d(this,typeof a==="function"?a:null,typeof b==="function"?b:null,c)};
Ud.prototype.$goog_Thenable=!0;n=Ud.prototype;n.qc=function(a,b){return $d(this,null,a,b)};
n.catch=Ud.prototype.qc;n.cancel=function(a){if(this.h==0){var b=new ae(a);Rd(function(){be(this,b)},this)}};
function be(a,b){if(a.h==0)if(a.j){var c=a.j;if(c.i){for(var d=0,e=null,f=null,g=c.i;g&&(g.j||(d++,g.child==a&&(e=g),!(e&&d>1)));g=g.next)e||(f=g);e&&(c.h==0&&d==1?be(c,b):(f?(d=f,d.next==c.l&&(c.l=d),d.next=d.next.next):ce(c),de(c,e,3,b)))}a.j=null}else Vd(a,3,b)}
function ee(a,b){a.i||a.h!=2&&a.h!=3||fe(a);a.l?a.l.next=b:a.i=b;a.l=b}
function $d(a,b,c,d){var e=Yd(null,null,null);e.child=new Ud(function(f,g){e.i=b?function(h){try{var k=b.call(d,h);f(k)}catch(l){g(l)}}:f;
e.h=c?function(h){try{var k=c.call(d,h);k===void 0&&h instanceof ae?g(h):f(k)}catch(l){g(l)}}:g});
e.child.j=a;ee(a,e);return e.child}
n.jf=function(a){this.h=0;Vd(this,2,a)};
n.kf=function(a){this.h=0;Vd(this,3,a)};
function Vd(a,b,c){if(a.h==0){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.h=1;a:{var d=c,e=a.jf,f=a.kf;if(d instanceof Ud){ee(d,Yd(e||Bd,f||null,a));var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(l){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Ra(d))try{var k=d.then;if(typeof k==="function"){ge(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}}g||(a.A=c,a.h=b,a.j=null,fe(a),b!=3||c instanceof ae||he(a,c))}}
function ge(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function fe(a){a.m||(a.m=!0,Rd(a.Wd,a))}
function ce(a){var b=null;a.i&&(b=a.i,a.i=b.next,b.next=null);a.i||(a.l=null);return b}
n.Wd=function(){for(var a;a=ce(this);)de(this,a,this.h,this.A);this.m=!1};
function de(a,b,c,d){if(c==3&&b.h&&!b.j)for(;a&&a.v;a=a.j)a.v=!1;if(b.child)b.child.j=null,ie(b,c,d);else try{b.j?b.i.call(b.context):ie(b,c,d)}catch(e){je.call(null,e)}Ad(Xd,b)}
function ie(a,b,c){b==2?a.i.call(a.context,c):a.h&&a.h.call(a.context,c)}
function he(a,b){a.v=!0;Rd(function(){a.v&&je.call(null,b)})}
var je=Kd;function ae(a){bb.call(this,a)}
$a(ae,bb);ae.prototype.name="cancel";function ke(a,b){wd.call(this);this.j=a||1;this.i=b||C;this.l=Xa(this.ff,this);this.m=Za()}
$a(ke,wd);n=ke.prototype;n.enabled=!1;n.Fa=null;n.setInterval=function(a){this.j=a;this.Fa&&this.enabled?(this.stop(),this.start()):this.Fa&&this.stop()};
n.ff=function(){if(this.enabled){var a=Za()-this.m;a>0&&a<this.j*.8?this.Fa=this.i.setTimeout(this.l,this.j-a):(this.Fa&&(this.i.clearTimeout(this.Fa),this.Fa=null),xd(this,"tick"),this.enabled&&(this.stop(),this.start()))}};
n.start=function(){this.enabled=!0;this.Fa||(this.Fa=this.i.setTimeout(this.l,this.j),this.m=Za())};
n.stop=function(){this.enabled=!1;this.Fa&&(this.i.clearTimeout(this.Fa),this.Fa=null)};
n.U=function(){ke.Ba.U.call(this);this.stop();delete this.i};
function le(a,b,c){if(typeof a==="function")c&&(a=Xa(a,c));else if(a&&typeof a.handleEvent=="function")a=Xa(a.handleEvent,a);else throw Error("Invalid listener argument");return Number(b)>2147483647?-1:C.setTimeout(a,b||0)}
;function me(a){G.call(this);this.H=a;this.j=0;this.l=100;this.m=!1;this.i=new Map;this.A=new Set;this.flushInterval=3E4;this.h=new ke(this.flushInterval);this.h.listen("tick",this.Aa,!1,this);Ec(this,this.h)}
y(me,G);n=me.prototype;n.sendIsolatedPayload=function(a){this.m=a;this.l=1};
function ne(a){a.h.enabled||a.h.start();a.j++;a.j>=a.l&&a.Aa()}
n.Aa=function(){var a=this.i.values();a=[].concat(la(a)).filter(function(b){return b.h.size});
a.length&&this.H.flush(a,this.m);oe(a);this.j=0;this.h.enabled&&this.h.stop()};
n.Ra=function(a){var b=B.apply(1,arguments);this.i.has(a)||this.i.set(a,new Ac(a,b))};
n.Eb=function(a){var b=B.apply(1,arguments);this.i.has(a)||this.i.set(a,new Bc(a,b))};
function pe(a,b){return a.A.has(b)?void 0:a.i.get(b)}
n.Ab=function(a){this.Id(a,1,B.apply(1,arguments))};
n.Id=function(a,b){var c=B.apply(2,arguments),d=pe(this,a);d&&d instanceof Ac&&(d.i(b,c),ne(this))};
n.record=function(a,b){var c=B.apply(2,arguments),d=pe(this,a);d&&d instanceof Bc&&(d.record(b,c),ne(this))};
function oe(a){for(var b=0;b<a.length;b++)a[b].clear()}
;function qe(a){this.h=a;this.h.Ra("/client_streamz/bg/fic",{G:3,F:"ke"})}
function re(a){this.h=a;this.h.Ra("/client_streamz/bg/fiec",{G:3,F:"rk"},{G:3,F:"ke"},{G:2,F:"ec"})}
function se(a){this.h=a;this.h.Eb("/client_streamz/bg/fil",{G:3,F:"rk"},{G:3,F:"ke"})}
se.prototype.record=function(a,b,c){this.h.record("/client_streamz/bg/fil",a,b,c)};
function te(a){this.h=a;this.h.Ra("/client_streamz/bg/fcc",{G:2,F:"ph"},{G:3,F:"ke"})}
function ue(a){this.h=a;this.h.Eb("/client_streamz/bg/fcd",{G:2,F:"ph"},{G:3,F:"ke"})}
ue.prototype.record=function(a,b,c){this.h.record("/client_streamz/bg/fcd",a,b,c)};
function ve(a){this.h=a;this.h.Ra("/client_streamz/bg/fsc",{G:3,F:"rk"},{G:3,F:"ke"})}
function we(a){this.h=a;this.h.Eb("/client_streamz/bg/fsl",{G:3,F:"rk"},{G:3,F:"ke"})}
we.prototype.record=function(a,b,c){this.h.record("/client_streamz/bg/fsl",a,b,c)};
function xe(a){this.h=a;this.h.Eb("/client_streamz/bg/wrl",{G:3,F:"mn"},{G:2,F:"ac"},{G:2,F:"sc"},{G:3,F:"rk"},{G:3,F:"mk"})}
xe.prototype.record=function(a,b,c,d,e,f){this.h.record("/client_streamz/bg/wrl",a,b,c,d,e,f)};
function ye(a){this.h=a;this.h.Eb("/client_streamz/bg/el",{G:3,F:"en"},{G:3,F:"bk"},{G:3,F:"rk"},{G:3,F:"mk"})}
ye.prototype.record=function(a,b,c,d,e){this.h.record("/client_streamz/bg/el",a,b,c,d,e)};
function ze(a){this.h=a;this.h.Ra("/client_streamz/bg/cec",{G:2,F:"ec"},{G:3,F:"bk"},{G:3,F:"rk"},{G:3,F:"mk"})}
function Ae(a){this.h=a;this.h.Ra("/client_streamz/bg/po/csc",{G:2,F:"cs"},{G:3,F:"rk"},{G:3,F:"mk"})}
function Be(a){this.h=a;this.h.Ra("/client_streamz/bg/po/ctav",{G:3,F:"av"},{G:3,F:"rk"},{G:3,F:"mk"})}
function Ce(a){this.h=a;this.h.Ra("/client_streamz/bg/po/cwsc",{G:3,F:"su"},{G:3,F:"rk"},{G:3,F:"mk"})}
;Pc();var De=Sc()||H("iPod"),Ee=H("iPad");!H("Android")||Qc()||Pc()||Oc()||H("Silk");Qc();var Fe=H("Safari")&&!(Qc()||(Nc()?0:H("Coast"))||Oc()||(Nc()?0:H("Edge"))||(Nc()?Mc("Microsoft Edge"):H("Edg/"))||(Nc()?Mc("Opera"):H("OPR"))||Pc()||H("Silk")||H("Android"))&&!(Sc()||H("iPad")||H("iPod"));var Ge={},He=null;function Ie(a,b){Qa(a);b===void 0&&(b=0);Je();b=Ge[b];for(var c=Array(Math.floor(a.length/3)),d=b[64]||"",e=0,f=0;e<a.length-2;e+=3){var g=a[e],h=a[e+1],k=a[e+2],l=b[g>>2];g=b[(g&3)<<4|h>>4];h=b[(h&15)<<2|k>>6];k=b[k&63];c[f++]=""+l+g+h+k}l=0;k=d;switch(a.length-e){case 2:l=a[e+1],k=b[(l&15)<<2]||d;case 1:a=a[e],c[f]=""+b[a>>2]+b[(a&3)<<4|l>>4]+k+d}return c.join("")}
function Ke(a){var b=a.length,c=b*3/4;c%3?c=Math.floor(c):"=.".indexOf(a[b-1])!=-1&&(c="=.".indexOf(a[b-2])!=-1?c-2:c-1);var d=new Uint8Array(c),e=0;Le(a,function(f){d[e++]=f});
return e!==c?d.subarray(0,e):d}
function Le(a,b){function c(k){for(;d<a.length;){var l=a.charAt(d++),m=He[l];if(m!=null)return m;if(!/^[\s\xa0]*$/.test(l))throw Error("Unknown base64 encoding at char: "+l);}return k}
Je();for(var d=0;;){var e=c(-1),f=c(0),g=c(64),h=c(64);if(h===64&&e===-1)break;b(e<<2|f>>4);g!=64&&(b(f<<4&240|g>>2),h!=64&&b(g<<6&192|h))}}
function Je(){if(!He){He={};for(var a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),b=["+/=","+/","-_=","-_.","-_"],c=0;c<5;c++){var d=a.concat(b[c].split(""));Ge[c]=d;for(var e=0;e<d.length;e++){var f=d[e];He[f]===void 0&&(He[f]=e)}}}}
;var Me=typeof Uint8Array!=="undefined",Ne=!Vc&&typeof btoa==="function";function Oe(a){if(!Ne)return Ie(a);for(var b="",c=0,d=a.length-10240;c<d;)b+=String.fromCharCode.apply(null,a.subarray(c,c+=10240));b+=String.fromCharCode.apply(null,c?a.subarray(c):a);return btoa(b)}
var Pe=/[-_.]/g,Qe={"-":"+",_:"/",".":"="};function Re(a){return Qe[a]||""}
function Se(a){return Me&&a!=null&&a instanceof Uint8Array}
var Te={};var Ue;function Ve(a){if(a!==Te)throw Error("illegal external caller");}
function We(a,b){Ve(b);this.h=a;if(a!=null&&a.length===0)throw Error("ByteString should be constructed with non-empty values");}
We.prototype.sizeBytes=function(){Ve(Te);var a=this.h;if(a!=null&&!Se(a))if(typeof a==="string")if(Ne){Pe.test(a)&&(a=a.replace(Pe,Re));a=atob(a);for(var b=new Uint8Array(a.length),c=0;c<a.length;c++)b[c]=a.charCodeAt(c);a=b}else a=Ke(a);else Pa(a),a=null;return(a=a==null?a:this.h=a)?a.length:0};function Xe(){return typeof BigInt==="function"}
;var Ye=0,Ze=0;function $e(a){var b=a<0;a=Math.abs(a);var c=a>>>0;a=Math.floor((a-c)/4294967296);b&&(c=x(af(c,a)),b=c.next().value,a=c.next().value,c=b);Ye=c>>>0;Ze=a>>>0}
function bf(a,b){b>>>=0;a>>>=0;if(b<=2097151)var c=""+(4294967296*b+a);else Xe()?c=""+(BigInt(b)<<BigInt(32)|BigInt(a)):(c=(a>>>24|b<<8)&16777215,b=b>>16&65535,a=(a&16777215)+c*6777216+b*6710656,c+=b*8147497,b*=2,a>=1E7&&(c+=Math.floor(a/1E7),a%=1E7),c>=1E7&&(b+=Math.floor(c/1E7),c%=1E7),c=b+cf(c)+cf(a));return c}
function cf(a){a=String(a);return"0000000".slice(a.length)+a}
function df(){var a=Ye,b=Ze;b&2147483648?Xe()?a=""+(BigInt(b|0)<<BigInt(32)|BigInt(a>>>0)):(b=x(af(a,b)),a=b.next().value,b=b.next().value,a="-"+bf(a,b)):a=bf(a,b);return a}
function af(a,b){b=~b;a?a=~a+1:b+=1;return[a,b]}
;function ef(a){return Array.prototype.slice.call(a)}
;var ff=typeof Symbol==="function"&&typeof Symbol()==="symbol";function gf(a){return typeof Symbol==="function"&&typeof Symbol()==="symbol"?Symbol():a}
var hf=gf(),jf=gf("0di"),kf=gf("2ex");function lf(a){return a.cf||(a.cf=gf("o_"+a[0]))}
;Math.max.apply(Math,la(Object.values({vg:1,tg:2,sg:4,yg:8,xg:16,wg:32,Af:64,Ag:128,rg:256,qg:512,ug:1024,Ff:2048,zg:4096,Gf:8192})));var mf=ff?function(a,b){a[hf]|=b}:function(a,b){a.Ua!==void 0?a.Ua|=b:Object.defineProperties(a,{Ua:{value:b,
configurable:!0,writable:!0,enumerable:!1}})};
function nf(a,b,c){return c?a|b:a&~b}
var of=ff?function(a){return a[hf]|0}:function(a){return a.Ua|0},pf=ff?function(a){return a[hf]}:function(a){return a.Ua},qf=hf?function(a,b){a[hf]=b;
return a}:function(a,b){a.Ua!==void 0?a.Ua=b:Object.defineProperties(a,{Ua:{value:b,
configurable:!0,writable:!0,enumerable:!1}});return a};
function rf(a){mf(a,34);return a}
function sf(a,b){qf(b,(a|0)&-14591)}
function tf(a,b){qf(b,(a|34)&-14557)}
function uf(a){a=a>>14&1023;return a===0?536870912:a}
;var vf={},wf={};function xf(a){return!(!a||typeof a!=="object"||a.re!==wf)}
function yf(a){return a!==null&&typeof a==="object"&&!Array.isArray(a)&&a.constructor===Object}
var zf;function Af(a,b,c){if(!Array.isArray(a)||a.length)return!1;var d=of(a);if(d&1)return!0;if(!(b&&(Array.isArray(b)?b.includes(c):b.has(c))))return!1;qf(a,d|1);return!0}
var Bf,Cf=[];qf(Cf,55);Bf=Object.freeze(Cf);function Df(a){if(a&2)throw Error();}
function Ef(a,b,c){this.j=0;this.h=a;this.i=b;this.thisArg=c}
Ef.prototype.next=function(){if(this.j<this.h.length){var a=this.h[this.j++];return{done:!1,value:this.i?this.i.call(this.thisArg,a):a}}return{done:!0,value:void 0}};
Ef.prototype[Symbol.iterator]=function(){return new Ef(this.h,this.i,this.thisArg)};
function Ff(){}
Object.freeze(new function(){});
Object.freeze(new Ff);var Gf=Object.freeze(new Ff);var Hf;function If(a){a=Error(a);ec(a,"warning");return a}
;function Jf(a){return a.displayName||a.name||"unknown type name"}
function Kf(a){if(a!=null&&typeof a!=="boolean")throw Error("Expected boolean but got "+Pa(a)+": "+a);return a}
var Lf=/^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;function Mf(a){var b=typeof a;return b==="number"?Number.isFinite(a):b!=="string"?!1:Lf.test(a)}
function Nf(a){if(typeof a!=="number")throw If("int32");if(!Number.isFinite(a))throw If("int32");return a|0}
function Of(a){return a==null?a:Nf(a)}
function Pf(a){if(a==null)return a;if(typeof a==="string"){if(!a)return;a=+a}if(typeof a==="number")return Number.isFinite(a)?a|0:void 0}
function Qf(a){if(a!=null){var b=!!b;if(!Mf(a))throw If("int64");a=typeof a==="string"?Rf(a):b?Sf(a):Tf(a)}return a}
function Uf(a){return a[0]==="-"?a.length<20?!0:a.length===20&&Number(a.substring(0,7))>-922337:a.length<19?!0:a.length===19&&Number(a.substring(0,6))<922337}
function Tf(a){Mf(a);a=Math.trunc(a);if(!Number.isSafeInteger(a)){$e(a);var b=Ye,c=Ze;if(a=c&2147483648)b=~b+1>>>0,c=~c>>>0,b==0&&(c=c+1>>>0);b=c*4294967296+(b>>>0);a=a?-b:b}return a}
function Sf(a){Mf(a);a=Math.trunc(a);if(Number.isSafeInteger(a))a=String(a);else{var b=String(a);Uf(b)?a=b:($e(a),a=df())}return a}
function Rf(a){Mf(a);var b=Math.trunc(Number(a));if(Number.isSafeInteger(b))return String(b);b=a.indexOf(".");b!==-1&&(a=a.substring(0,b));a.indexOf(".");if(!Uf(a)){if(a.length<16)$e(Number(a));else if(Xe())a=BigInt(a),Ye=Number(a&BigInt(4294967295))>>>0,Ze=Number(a>>BigInt(32)&BigInt(4294967295));else{b=+(a[0]==="-");Ze=Ye=0;for(var c=a.length,d=0+b,e=(c-b)%6+b;e<=c;d=e,e+=6)d=Number(a.slice(d,e)),Ze*=1E6,Ye=Ye*1E6+d,Ye>=4294967296&&(Ze+=Math.trunc(Ye/4294967296),Ze>>>=0,Ye>>>=0);b&&(b=x(af(Ye,Ze)),
a=b.next().value,b=b.next().value,Ye=a,Ze=b)}a=df()}return a}
function Vf(a){if(typeof a!=="string")throw Error();return a}
function Wf(a){if(a!=null&&typeof a!=="string")throw Error();return a}
function Xf(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+Jf(b)+" but got "+(a&&Jf(a.constructor)));}
function Yf(a,b,c,d){if(a!=null&&typeof a==="object"&&a.Fc===vf)return a;if(!Array.isArray(a))return c?d&2?(a=b[jf])?b=a:(a=new b,rf(a.D),b=b[jf]=a):b=new b:b=void 0,b;var e=c=of(a);e===0&&(e|=d&32);e|=d&2;e!==c&&qf(a,e);return new b(a)}
;var Zf;function $f(a,b){of(b);Zf=b;a=new a(b);Zf=void 0;return a}
function I(a,b,c){a==null&&(a=Zf);Zf=void 0;if(a==null){var d=96;c?(a=[c],d|=512):a=[];b&&(d=d&-16760833|(b&1023)<<14)}else{if(!Array.isArray(a))throw Error("narr");d=of(a);if(d&2048)throw Error("farr");if(d&64)return a;d|=64;if(c&&(d|=512,c!==a[0]))throw Error("mid");a:{c=a;var e=c.length;if(e){var f=e-1;if(yf(c[f])){d|=256;b=f-(+!!(d&512)-1);if(b>=1024)throw Error("pvtlmt");d=d&-16760833|(b&1023)<<14;break a}}if(b){b=Math.max(b,e-(+!!(d&512)-1));if(b>1024)throw Error("spvt");d=d&-16760833|(b&1023)<<
14}}}qf(a,d);return a}
;var ag=function(){try{var a=function(){return qa(Map,[],this.constructor)};
y(a,Map);Tc(new a);return!1}catch(b){return!0}}();
function bg(){this.h=new Map}
n=bg.prototype;n.get=function(a){return this.h.get(a)};
n.set=function(a,b){this.h.set(a,b);this.size=this.h.size;return this};
n.delete=function(a){a=this.h.delete(a);this.size=this.h.size;return a};
n.clear=function(){this.h.clear();this.size=this.h.size};
n.has=function(a){return this.h.has(a)};
n.entries=function(){return this.h.entries()};
n.keys=function(){return this.h.keys()};
n.values=function(){return this.h.values()};
n.forEach=function(a,b){return this.h.forEach(a,b)};
bg.prototype[Symbol.iterator]=function(){return this.entries()};
var cg=function(){function a(){return qa(Map,[],this.constructor)}
if(ag)return Object.setPrototypeOf(bg.prototype,Map.prototype),Object.defineProperties(bg.prototype,{size:{value:0,configurable:!0,enumerable:!0,writable:!0}}),bg;y(a,Map);return a}();
function dg(a){return a}
function eg(a,b,c,d){c=c===void 0?dg:c;d=d===void 0?dg:d;var e=cg.call(this)||this;var f=of(a);f|=64;qf(a,f);e.Ub=f;e.sc=b;e.Kb=c;e.Pc=e.sc?fg:d;for(var g=0;g<a.length;g++){var h=a[g],k=c(h[0],!1,!0),l=h[1];b?l===void 0&&(l=null):l=d(h[1],!1,!0,void 0,void 0,f);cg.prototype.set.call(e,k,l)}return e}
y(eg,cg);function gg(a){if(a.Ub&2)throw Error("Cannot mutate an immutable Map");}
function hg(a,b){b=b===void 0?ig:b;if(a.size!==0)return jg(a,b)}
function jg(a,b){b=b===void 0?ig:b;var c=[];a=cg.prototype.entries.call(a);for(var d;!(d=a.next()).done;)d=d.value,d[0]=b(d[0]),d[1]=b(d[1]),c.push(d);return c}
n=eg.prototype;n.clear=function(){gg(this);cg.prototype.clear.call(this)};
n.delete=function(a){gg(this);return cg.prototype.delete.call(this,this.Kb(a,!0,!1))};
n.entries=function(){var a=Array.from(cg.prototype.keys.call(this));return new Ef(a,kg,this)};
n.keys=function(){return cg.prototype.keys.call(this)};
n.values=function(){var a=Array.from(cg.prototype.keys.call(this));return new Ef(a,eg.prototype.get,this)};
n.forEach=function(a,b){var c=this;cg.prototype.forEach.call(this,function(d,e){a.call(b,c.get(e),e,c)})};
n.set=function(a,b){gg(this);a=this.Kb(a,!0,!1);return a==null?this:b==null?(cg.prototype.delete.call(this,a),this):cg.prototype.set.call(this,a,this.Pc(b,!0,!0,this.sc,!1,this.Ub))};
n.has=function(a){return cg.prototype.has.call(this,this.Kb(a,!1,!1))};
n.get=function(a){a=this.Kb(a,!1,!1);var b=cg.prototype.get.call(this,a);if(b!==void 0){var c=this.sc;return c?(c=this.Pc(b,!1,!0,c,this.Fg,this.Ub),c!==b&&cg.prototype.set.call(this,a,c),c):b}};
eg.prototype[Symbol.iterator]=function(){return this.entries()};
eg.prototype.toJSON=void 0;eg.prototype.re=wf;function fg(a,b,c,d,e,f){b&&Xf(a,d);a=Yf(a,d,c,f);e&&(a=lg(a));f&2&&of(a.D);return a}
function ig(a){return a}
function kg(a){return[a,this.get(a)]}
;function mg(a,b){return ng(b)}
function ng(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "boolean":return a?1:0;case "object":if(a)if(Array.isArray(a)){if(Af(a,void 0,0))return}else{if(Se(a))return Oe(a);if(a instanceof We){var b=a.h;return b==null?"":typeof b==="string"?b:a.h=Oe(b)}if(a instanceof eg)return hg(a)}}return a}
;function og(a,b,c){a=ef(a);var d=a.length,e=b&256?a[d-1]:void 0;d+=e?-1:0;for(b=b&512?1:0;b<d;b++)a[b]=c(a[b]);if(e){b=a[b]={};for(var f in e)b[f]=c(e[f])}return a}
function pg(a,b,c,d,e){if(a!=null){if(Array.isArray(a))a=Af(a,void 0,0)?void 0:e&&of(a)&2?a:qg(a,b,c,d!==void 0,e);else if(yf(a)){var f={},g;for(g in a)f[g]=pg(a[g],b,c,d,e);a=f}else a=b(a,d);return a}}
function qg(a,b,c,d,e){var f=d||c?of(a):0;d=d?!!(f&32):void 0;a=ef(a);for(var g=0;g<a.length;g++)a[g]=pg(a[g],b,c,d,e);c&&c(f,a);return a}
function rg(a){return pg(a,sg,void 0,void 0,!1)}
function sg(a){if(a.Fc===vf){var b=b===void 0?tg:b;ug(b);a=a.toJSON()}else a=a instanceof eg?hg(a,rg):ng(a);return a}
;function vg(a,b,c){c=c===void 0?tf:c;if(a!=null){if(Me&&a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){var d=of(a);d&2||(b&&(b=d===0||!!(d&32)&&!(d&64||!(d&16))),a=b?qf(a,(d|34)&-12293):qg(a,vg,d&4?tf:c,!0,!0));return a}a.Fc===vf?(c=a.D,d=pf(c),a=d&2?a:$f(a.constructor,wg(c,d,!0))):a instanceof eg&&!(a.Ub&2)&&(c=rf(jg(a,vg)),a=new eg(c,a.sc,a.Kb,a.Pc));return a}}
function wg(a,b,c){var d=c||b&2?tf:sf,e=!!(b&32);a=og(a,b,function(f){return vg(f,e,d)});
mf(a,32|(c?2:0));return a}
function lg(a){var b=a.D,c=pf(b);return c&2?$f(a.constructor,wg(b,c,!1)):a}
;function xg(a,b){a=a.D;return yg(a,pf(a),b)}
function zg(a,b,c,d){b=d+(+!!(b&512)-1);if(!(b<0||b>=a.length||b>=c))return a[b]}
function yg(a,b,c,d){if(c===-1)return null;var e=uf(b);if(c>=e){if(b&256)return a[a.length-1][c]}else{var f=a.length;if(d&&b&256&&(d=a[f-1][c],d!=null)){if(zg(a,b,e,c)&&kf!=null){var g;a=(g=Hf)!=null?g:Hf={};g=a[kf]||0;g>=4||(a[kf]=g+1,g=Error(),ec(g,"incident"),Kd(g))}return d}return zg(a,b,e,c)}}
function K(a,b,c){var d=a.D,e=pf(d);Df(e);Ag(d,e,b,c);return a}
function Ag(a,b,c,d,e){yf(d);var f=uf(b);if(c>=f||e){var g=b;if(b&256)e=a[a.length-1];else{if(d==null)return g;e=a[f+(+!!(b&512)-1)]={};g|=256}e[c]=d;c<f&&(a[c+(+!!(b&512)-1)]=void 0);g!==b&&qf(a,g);return g}a[c+(+!!(b&512)-1)]=d;b&256&&(a=a[a.length-1],c in a&&delete a[c]);return b}
function Bg(a){return Cg(a,Dg,11,!1)!==void 0}
function Eg(a){return!!(2&a)&&!!(4&a)||!!(2048&a)}
function Fg(a,b,c){var d=a.D,e=pf(d);Df(e);if(b==null)return Ag(d,e,3),a;if(!Array.isArray(b))throw If();var f=of(b),g=f,h=!!(2&f)||Object.isFrozen(b),k=!h&&(void 0===Gf||!1);if(!(4&f))for(f=21,h&&(b=ef(b),g=0,f=Gg(f,e),f=Hg(f,e,!0)),h=0;h<b.length;h++)b[h]=c(b[h]);k&&(b=ef(b),g=0,f=Gg(f,e),f=Hg(f,e,!0));f!==g&&qf(b,f);Ag(d,e,3,b);return a}
function Ig(a,b,c,d){a=a.D;var e=pf(a);Df(e);var f=e;var g=lf(c),h=a[g];if(h==null){h=0;for(var k=c.length-1;k>=0;k--){var l=c[k];h===0&&yg(a,f,l)!=null?h=l:h!==0&&(f=Ag(a,f,l))}Jg(g,a,h)}f=h;if(d==null)if(f===b)Jg(lf(c),a,0);else return;else f!==b&&(f&&(e=Ag(a,e,f)),Jg(lf(c),a,b));Ag(a,e,b,d)}
function Jg(a,b,c){ff||a in b?b[a]=c:Object.defineProperty(b,a,{value:c,writable:!0})}
function Cg(a,b,c,d){a=a.D;var e=pf(a),f=yg(a,e,c,d);b=Yf(f,b,!1,e);b!==f&&b!=null&&Ag(a,e,c,b,d);return b}
function Kg(a,b,c,d){d=d===void 0?!1:d;b=Cg(a,b,c,d);if(b==null)return b;a=a.D;var e=pf(a);if(!(e&2)){var f=lg(b);f!==b&&(b=f,Ag(a,e,c,b,d))}return b}
function Lg(a,b,c,d){d!=null?Xf(d,b):d=void 0;return K(a,c,d)}
function Mg(a,b,c,d){var e=a.D,f=pf(e);Df(f);if(d==null)return Ag(e,f,c),a;if(!Array.isArray(d))throw If();for(var g=of(d),h=g,k=!!(2&g)||!!(2048&g),l=k||Object.isFrozen(d),m=!l&&(void 0===Gf||!1),p=!0,t=!0,r=0;r<d.length;r++){var v=d[r];Xf(v,b);k||(v=!!(of(v.D)&2),p&&(p=!v),t&&(t=v))}k||(g=nf(g,5,!0),g=nf(g,8,p),g=nf(g,16,t));if(m||l&&g!==h)d=ef(d),h=0,g=Gg(g,f),g=Hg(g,f,!0);g!==h&&qf(d,g);Ag(e,f,c,d);return a}
function Gg(a,b){a=nf(a,2,!!(2&b));a=nf(a,32,!0);return a=nf(a,2048,!1)}
function Hg(a,b,c){32&b&&c||(a=nf(a,32,!1));return a}
function Ng(a,b){a=xg(a,b);var c;a==null?c=a:Mf(a)?typeof a==="number"?c=Tf(a):c=Rf(a):c=void 0;return c}
function Og(a){a=xg(a,1);var b=b===void 0?!1:b;b=a==null?a:Mf(a)?typeof a==="string"?Rf(a):b?Sf(a):Tf(a):void 0;return b}
function Pg(a){var b=0;b=b===void 0?0:b;a=xg(a,1);a=a==null?a:Number.isFinite(a)?a|0:void 0;return a!=null?a:b}
function Sg(a,b,c){return K(a,b,Wf(c))}
function Tg(a,b,c){if(c!=null){if(!Number.isFinite(c))throw If("enum");c|=0}return K(a,b,c)}
;function Ug(a){return a}
function Vg(a,b,c,d){return Wg(a,b,c,d,Xg,Yg)}
function Zg(a,b,c,d){return Wg(a,b,c,d,$g,ah)}
function Wg(a,b,c,d,e,f){if(!c.length&&!d)return 0;for(var g=0,h=0,k=0,l=0,m=0,p=c.length-1;p>=0;p--){var t=c[p];d&&p===c.length-1&&t===d||(l++,t!=null&&k++)}if(d)for(var r in d)p=+r,isNaN(p)||(m+=bh(p),h++,p>g&&(g=p));l=e(l,k)+f(h,g,m);r=k;p=h;t=g;for(var v=m,u=c.length-1;u>=0;u--){var z=c[u];if(!(z==null||d&&u===c.length-1&&z===d)){z=u-b;var F=e(z,r)+f(p,t,v);F<l&&(a=1+z,l=F);p++;r--;v+=bh(z);t=Math.max(t,z)}}b=e(0,0)+f(p,t,v);b<l&&(a=0,l=b);if(d){p=h;t=g;v=m;r=k;for(var J in d)d=+J,isNaN(d)||d>=
1024||(p--,r++,v-=J.length,g=e(d,r)+f(p,t,v),g<l&&(a=1+d,l=g))}return a}
function ah(a,b,c){return c+a*3+(a>1?a-1:0)}
function $g(a,b){return(a>1?a-1:0)+(a-b)*4}
function Yg(a,b){return a==0?0:9*Math.max(1<<32-Math.clz32(a+a/2-1),4)<=b?a==0?0:a<4?100+(a-1)*16:a<6?148+(a-4)*16:a<12?244+(a-6)*16:a<22?436+(a-12)*19:a<44?820+(a-22)*17:52+32*a:40+4*b}
function Xg(a){return 40+4*a}
function bh(a){return a>=100?a>=1E4?Math.ceil(Math.log10(1+a)):a<1E3?3:4:a<10?1:2}
function ug(a){switch(a){case Ug:case Vg:case Zg:break;default:throw Error("ipsel");}}
;var tg=Ug;function L(a,b,c){this.D=I(a,b,c)}
n=L.prototype;n.toJSON=function(){if(zf)var a=ch(this,this.D,!1);else a=qg(this.D,sg,void 0,void 0,!1),a=ch(this,a,!0);return a};
n.serialize=function(a){a=a===void 0?tg:a;zf=!0;try{return a!==Ug&&ug(a),JSON.stringify(this.toJSON(),mg)}finally{zf=!1}};
function dh(a,b){if(b==null||b=="")return new a;b=JSON.parse(b);if(!Array.isArray(b))throw Error("dnarr");mf(b,32);return $f(a,b)}
n.clone=function(){var a=this.D,b=pf(a);return $f(this.constructor,wg(a,b,!1))};
n.Fc=vf;n.toString=function(){return ch(this,this.D,!1).toString()};
function ch(a,b,c){var d=Ic?void 0:a.constructor.Qa;c=pf(c?a.D:b);a=b.length;if(!a)return b;var e=b[a-1],f=yf(e);f?a--:e=void 0;c=+!!(c&512)-1;var g=a-c,h=g!==g,k=h?Array.prototype.slice.call(b,0,a):b;if(f||h){a:{var l=k;var m=e;f={};var p=!1;if(h)for(var t=Math.max(0,g+c);t<l.length;t++){var r=l[t],v=t-c;r==null||Af(r,d,v)||xf(r)&&r.size===0||(l[t]=void 0,f[v]=r,p=!0)}if(m)for(var u in m)if(t=+u,isNaN(t))f[u]=m[u];else if(r=m[u],Array.isArray(r)&&(Af(r,d,+u)||xf(r)&&r.size===0)&&(r=null),r==null&&
(p=!0),t<g&&h){p=!0;r=t+c;for(v=l.length;v<=r;v++)l.push(void 0);l[r]=m[t]}else r!=null&&(f[u]=r);if(p){for(var z in f){m=f;break a}m=null}}l=m==null?e!=null:m!==e}h&&(a=k.length);for(var F;a>0;a--){z=a-1;u=k[z];z-=c;if(!(u==null||Af(u,d,z)||xf(u)&&u.size===0))break;F=!0}if(k===b&&!l&&!F)return k;h?F&&(k.length=a):k=Array.prototype.slice.call(k,0,a);m&&k.push(m);return k}
;Object.freeze({});function eh(a){a.Lg=!0;return a}
;function fh(a){this.D=I(a)}
y(fh,L);fh.Qa=[1,2,3,4];var gh={toString:function(a){var b=[],c=0;a-=-2147483648;b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(a%52);for(a=Math.floor(a/52);a>0;)b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".charAt(a%62),a=Math.floor(a/62);return b.join("")}};function hh(a){function b(){c-=d;c-=e;c^=e>>>13;d-=e;d-=c;d^=c<<8;e-=c;e-=d;e^=d>>>13;c-=d;c-=e;c^=e>>>12;d-=e;d-=c;d^=c<<16;e-=c;e-=d;e^=d>>>5;c-=d;c-=e;c^=e>>>3;d-=e;d-=c;d^=c<<10;e-=c;e-=d;e^=d>>>15}
a=ih(a);for(var c=2654435769,d=2654435769,e=314159265,f=a.length,g=f,h=0;g>=12;g-=12,h+=12)c+=jh(a,h),d+=jh(a,h+4),e+=jh(a,h+8),b();e+=f;switch(g){case 11:e+=a[h+10]<<24;case 10:e+=a[h+9]<<16;case 9:e+=a[h+8]<<8;case 8:d+=a[h+7]<<24;case 7:d+=a[h+6]<<16;case 6:d+=a[h+5]<<8;case 5:d+=a[h+4];case 4:c+=a[h+3]<<24;case 3:c+=a[h+2]<<16;case 2:c+=a[h+1]<<8;case 1:c+=a[h+0]}b();return gh.toString(e)}
function ih(a){for(var b=[],c=0;c<a.length;c++)b.push(a.charCodeAt(c));return b}
function jh(a,b){return a[b+0]+(a[b+1]<<8)+(a[b+2]<<16)+(a[b+3]<<24)}
;function kh(a){this.D=I(a)}
y(kh,L);var lh=[1,2,3];function mh(a){this.D=I(a)}
y(mh,L);var nh=[1,2,3];function oh(a){this.D=I(a)}
y(oh,L);oh.Qa=[1];function ph(a){this.D=I(a)}
y(ph,L);ph.Qa=[3,6,4];function qh(a){this.D=I(a)}
y(qh,L);qh.Qa=[1];function rh(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a.indexOf("blob:")===0&&(a=a.substring(5));a=a.split("#")[0].split("?")[0];a=a.toLowerCase();a.indexOf("//")==0&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");c!=-1&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if(c!=="http"&&c!=="https"&&c!=="chrome-extension"&&
c!=="moz-extension"&&c!=="file"&&c!=="android-app"&&c!=="chrome-search"&&c!=="chrome-untrusted"&&c!=="chrome"&&c!=="app"&&c!=="devtools")throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(d!=-1){var e=b.substring(d+1);b=b.substring(0,d);if(c==="http"&&e!=="80"||c==="https"&&e!=="443")a=":"+e}return c+"://"+b+a}
;function sh(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;m=l=0}
function b(p){for(var t=g,r=0;r<64;r+=4)t[r/4]=p[r]<<24|p[r+1]<<16|p[r+2]<<8|p[r+3];for(r=16;r<80;r++)p=t[r-3]^t[r-8]^t[r-14]^t[r-16],t[r]=(p<<1|p>>>31)&4294967295;p=e[0];var v=e[1],u=e[2],z=e[3],F=e[4];for(r=0;r<80;r++){if(r<40)if(r<20){var J=z^v&(u^z);var O=1518500249}else J=v^u^z,O=1859775393;else r<60?(J=v&u|z&(v|u),O=2400959708):(J=v^u^z,O=3395469782);J=((p<<5|p>>>27)&4294967295)+J+F+O+t[r]&4294967295;F=z;z=u;u=(v<<30|v>>>2)&4294967295;v=p;p=J}e[0]=e[0]+p&4294967295;e[1]=e[1]+v&4294967295;e[2]=
e[2]+u&4294967295;e[3]=e[3]+z&4294967295;e[4]=e[4]+F&4294967295}
function c(p,t){if(typeof p==="string"){p=unescape(encodeURIComponent(p));for(var r=[],v=0,u=p.length;v<u;++v)r.push(p.charCodeAt(v));p=r}t||(t=p.length);r=0;if(l==0)for(;r+64<t;)b(p.slice(r,r+64)),r+=64,m+=64;for(;r<t;)if(f[l++]=p[r++],m++,l==64)for(l=0,b(f);r+64<t;)b(p.slice(r,r+64)),r+=64,m+=64}
function d(){var p=[],t=m*8;l<56?c(h,56-l):c(h,64-(l-56));for(var r=63;r>=56;r--)f[r]=t&255,t>>>=8;b(f);for(r=t=0;r<5;r++)for(var v=24;v>=0;v-=8)p[t++]=e[r]>>v&255;return p}
for(var e=[],f=[],g=[],h=[128],k=1;k<64;++k)h[k]=0;var l,m;a();return{reset:a,update:c,digest:d,Sd:function(){for(var p=d(),t="",r=0;r<p.length;r++)t+="0123456789ABCDEF".charAt(Math.floor(p[r]/16))+"0123456789ABCDEF".charAt(p[r]%16);return t}}}
;function th(a,b,c){var d=String(C.location.href);return d&&a&&b?[b,uh(rh(d),a,c||null)].join(" "):null}
function uh(a,b,c){var d=[],e=[];if((Array.isArray(c)?2:1)==1)return e=[b,a],Db(d,function(h){e.push(h)}),vh(e.join(" "));
var f=[],g=[];Db(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=f.length==0?[c,b,a]:[f.join(":"),c,b,a];Db(d,function(h){e.push(h)});
a=vh(e.join(" "));a=[c,a];g.length==0||a.push(g.join(""));return a.join("_")}
function vh(a){var b=sh();b.update(a);return b.Sd().toLowerCase()}
;var wh={};function xh(a){this.h=a||{cookie:""}}
n=xh.prototype;n.isEnabled=function(){if(!C.navigator.cookieEnabled)return!1;if(this.h.cookie)return!0;this.set("TESTCOOKIESENABLED","1",{Nb:60});if(this.get("TESTCOOKIESENABLED")!=="1")return!1;this.remove("TESTCOOKIESENABLED");return!0};
n.set=function(a,b,c){var d=!1;if(typeof c==="object"){var e=c.Le;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.Nb}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');h===void 0&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=h<0?"":h==0?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+h*1E3)).toUTCString();this.h.cookie=a+"="+b+c+g+h+d+(e!=null?";samesite="+
e:"")};
n.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=db(d[e]);if(f.lastIndexOf(c,0)==0)return f.slice(c.length);if(f==a)return""}return b};
n.remove=function(a,b,c){var d=this.get(a)!==void 0;this.set(a,"",{Nb:0,path:b,domain:c});return d};
n.clear=function(){for(var a=(this.h.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=db(a[f]),d=e.indexOf("="),d==-1?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;a>=0;a--)this.remove(b[a])};
var yh=new xh(typeof document=="undefined"?null:document);function zh(a){return!!wh.FPA_SAMESITE_PHASE2_MOD||!(a===void 0||!a)}
function Ah(a){a=a===void 0?!1:a;var b=C.__SAPISID||C.__APISID||C.__3PSAPISID||C.__OVERRIDE_SID;zh(a)&&(b=b||C.__1PSAPISID);if(b)return!0;if(typeof document!=="undefined"){var c=new xh(document);b=c.get("SAPISID")||c.get("APISID")||c.get("__Secure-3PAPISID");zh(a)&&(b=b||c.get("__Secure-1PAPISID"))}return!!b}
function Bh(a,b,c,d){(a=C[a])||typeof document==="undefined"||(a=(new xh(document)).get(b));return a?th(a,c,d):null}
function Ch(a,b){b=b===void 0?!1:b;var c=rh(String(C.location.href)),d=[];if(Ah(b)){c=c.indexOf("https:")==0||c.indexOf("chrome-extension:")==0||c.indexOf("moz-extension:")==0;var e=c?C.__SAPISID:C.__APISID;e||typeof document==="undefined"||(e=new xh(document),e=e.get(c?"SAPISID":"APISID")||e.get("__Secure-3PAPISID"));(e=e?th(e,c?"SAPISIDHASH":"APISIDHASH",a):null)&&d.push(e);c&&zh(b)&&((b=Bh("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&d.push(b),(a=Bh("__3PSAPISID","__Secure-3PAPISID",
"SAPISID3PHASH",a))&&d.push(a))}return d.length==0?null:d.join(" ")}
;function Dh(a){this.D=I(a)}
y(Dh,L);Dh.Qa=[2];function Eh(a,b){this.intervalMs=a;this.callback=b;this.enabled=!1;this.h=function(){return Za()};
this.i=this.h()}
Eh.prototype.setInterval=function(a){this.intervalMs=a;this.timer&&this.enabled?(this.stop(),this.start()):this.timer&&this.stop()};
Eh.prototype.start=function(){var a=this;this.enabled=!0;this.timer||(this.timer=setTimeout(function(){a.tick()},this.intervalMs),this.i=this.h())};
Eh.prototype.stop=function(){this.enabled=!1;this.timer&&(clearTimeout(this.timer),this.timer=void 0)};
Eh.prototype.tick=function(){var a=this;if(this.enabled){var b=Math.max(this.h()-this.i,0);b<this.intervalMs*.8?this.timer=setTimeout(function(){a.tick()},this.intervalMs-b):(this.timer&&(clearTimeout(this.timer),this.timer=void 0),this.callback(),this.enabled&&(this.stop(),this.start()))}else this.timer=void 0};function Fh(a){this.D=I(a)}
y(Fh,L);function Gh(a){this.D=I(a)}
y(Gh,L);function Hh(a){this.h=this.i=this.j=a}
Hh.prototype.reset=function(){this.h=this.i=this.j};
Hh.prototype.getValue=function(){return this.i};function Ih(a){this.D=I(a)}
y(Ih,L);Ih.prototype.ec=function(){return Pg(this)};function Jh(a){this.D=I(a)}
y(Jh,L);function Kh(a){this.D=I(a)}
y(Kh,L);function Lh(a,b){Mg(a,Jh,1,b)}
Kh.Qa=[1];function Dg(a){this.D=I(a)}
y(Dg,L);var Mh=["platform","platformVersion","architecture","model","uaFullVersion"],Nh=new Kh,Oh=null;function Ph(a,b){b=b===void 0?Mh:b;if(!Oh){var c;a=(c=a.navigator)==null?void 0:c.userAgentData;if(!a||typeof a.getHighEntropyValues!=="function"||a.brands&&typeof a.brands.map!=="function")return Promise.reject(Error("UACH unavailable"));c=(a.brands||[]).map(function(e){var f=new Jh;f=Sg(f,1,e.brand);return Sg(f,2,e.version)});
Lh(K(Nh,2,Kf(a.mobile)),c);Oh=a.getHighEntropyValues(b)}var d=new Set(b);return Oh.then(function(e){var f=Nh.clone();d.has("platform")&&Sg(f,3,e.platform);d.has("platformVersion")&&Sg(f,4,e.platformVersion);d.has("architecture")&&Sg(f,5,e.architecture);d.has("model")&&Sg(f,6,e.model);d.has("uaFullVersion")&&Sg(f,7,e.uaFullVersion);return f}).catch(function(){return Nh.clone()})}
;function Qh(a){this.D=I(a)}
y(Qh,L);function Rh(a){this.D=I(a,4)}
y(Rh,L);function Sh(a){this.D=I(a,35)}
y(Sh,L);Sh.Qa=[3,20,27];function Th(a){this.D=I(a,19)}
y(Th,L);Th.prototype.Qb=function(a){return Tg(this,2,a)};
Th.Qa=[3,5];function Uh(a){this.D=I(a,8)}
y(Uh,L);var Vh=function(a){return function(b){return dh(a,b)}}(Uh);
Uh.Qa=[5,6,7];function Wh(a){this.D=I(a)}
y(Wh,L);var Xh;Xh=new function(a,b){this.h=a;this.ctor=b;this.isRepeated=0;this.i=Kg;this.defaultValue=void 0}(175237375,Wh);function Yh(a){G.call(this);var b=this;this.componentId="";this.j=[];this.da="";this.pageId=null;this.ga=this.W=-1;this.experimentIds=null;this.P=this.m=0;this.ia=1;this.timeoutMillis=0;this.logSource=a.logSource;this.Jb=a.Jb||function(){};
this.i=new Zh(a.logSource,a.eb);this.network=a.network;this.yb=a.yb||null;this.bufferSize=1E3;this.A=a.lf||null;this.sessionIndex=a.sessionIndex||null;this.Hb=a.Hb||!1;this.logger=null;this.withCredentials=!a.Zc;this.eb=a.eb||!1;this.H=typeof URLSearchParams!=="undefined"&&!!(new URL($h())).searchParams&&!!(new URL($h())).searchParams.set;var c=Tg(new Qh,1,1);ai(this.i,c);this.l=new Hh(1E4);a=bi(this,a.Tc);this.h=new Eh(this.l.getValue(),a);this.ba=new Eh(6E5,a);this.Hb||this.ba.start();this.eb||
(document.addEventListener("visibilitychange",function(){document.visibilityState==="hidden"&&b.xc()}),document.addEventListener("pagehide",this.xc.bind(this)))}
y(Yh,G);function bi(a,b){return a.H?b?function(){b().then(function(){a.flush()})}:function(){a.flush()}:function(){}}
n=Yh.prototype;n.U=function(){this.xc();this.h.stop();this.ba.stop();G.prototype.U.call(this)};
n.log=function(a){if(this.H){a=a.clone();var b=this.ia++;a=K(a,21,Qf(b));this.componentId&&Sg(a,26,this.componentId);if(Og(a)==null){var c=Date.now();b=a;c=Number.isFinite(c)?c.toString():"0";K(b,1,Qf(c))}Ng(a,15)==null&&K(a,15,Qf((new Date).getTimezoneOffset()*60));this.experimentIds&&(b=a,c=this.experimentIds.clone(),Lg(b,Dh,16,c));b=this.j.length-this.bufferSize+1;b>0&&(this.j.splice(0,b),this.m+=b);this.j.push(a);this.Hb||this.h.enabled||this.h.start()}};
n.flush=function(a,b){var c=this;if(this.j.length===0)a&&a();else{var d=Date.now();if(this.ga>d&&this.W<d)b&&b("throttled");else{this.network&&(typeof this.network.ec==="function"?ci(this.i,this.network.ec()):ci(this.i,0));var e=di(this.i,this.j,this.m,this.P,this.yb);d={};var f=this.Jb();f&&(d.Authorization=f);this.A||(this.A=$h());try{var g=(new URL(this.A)).toString()}catch(k){g=(new URL(this.A,window.location.origin)).toString()}g=new URL(g);this.sessionIndex&&(d["X-Goog-AuthUser"]=this.sessionIndex,
g.searchParams.set("authuser",this.sessionIndex));this.pageId&&(Object.defineProperty(d,"X-Goog-PageId",{value:this.pageId}),g.searchParams.set("pageId",this.pageId));if(f&&this.da===f)b&&b("stale-auth-token");else{this.j=[];this.h.enabled&&this.h.stop();this.m=0;var h=e.serialize();d={url:g.toString(),body:h,Dg:1,sd:d,requestType:"POST",withCredentials:this.withCredentials,timeoutMillis:this.timeoutMillis};g=function(k){c.l.reset();c.h.setInterval(c.l.getValue());if(k){var l=null;try{var m=JSON.stringify(JSON.parse(k.replace(")]}'\n",
"")));l=Vh(m)}catch(t){}if(l){k=Number;m="-1";m=m===void 0?"0":m;var p=Og(l);k=k(p!=null?p:m);k>0&&(c.W=Date.now(),c.ga=c.W+k);l=Xh.ctor?Xh.i(l,Xh.ctor,Xh.h,!0):Xh.i(l,Xh.h,null,!0);if(k=l===null?void 0:l)l=-1,l=l===void 0?0:l,k=Pf(xg(k,1)),l=k!=null?k:l,l!==-1&&(c.l=new Hh(l<1?1:l),c.h.setInterval(c.l.getValue()))}}a&&a();c.P=0};
h=function(k,l){var m=e.D;var p=pf(m),t=p,r=!(2&p),v=!!(2&t);p=v?1:2;r&&(r=!v);v=yg(m,t,3);v=Array.isArray(v)?v:Bf;var u=of(v),z=!!(4&u);if(!z){var F=u;F===0&&(F=Gg(F,t));F=nf(F,1,!0);u=v;var J=t,O=!!(2&F);O&&(J=nf(J,2,!0));for(var S=!O,da=!0,va=0,P=0;va<u.length;va++){var ea=Yf(u[va],Sh,!1,J);if(ea instanceof Sh){if(!O){var na=!!(of(ea.D)&2);S&&(S=!na);da&&(da=na)}u[P++]=ea}}P<va&&(u.length=P);F=nf(F,4,!0);F=nf(F,16,da);F=nf(F,8,S);qf(u,F);O&&Object.freeze(u);u=F}if(r&&!(8&u||!v.length&&(p===1||
p===4&&32&u))){Eg(u)&&(v=ef(v),u=Gg(u,t),t=Ag(m,t,3,v));r=v;for(F=0;F<r.length;F++)J=r[F],O=lg(J),J!==O&&(r[F]=O);u=nf(u,8,!0);u=nf(u,16,!r.length);qf(r,u)}Eg(u)||(r=u,(F=p===1||p===4&&!!(32&u))?(J=!!(32&u),u=nf(u,!v.length||16&u&&(!z||J)?2:2048,!0)):u=Hg(u,t,!1),u!==r&&qf(v,u),F&&Object.freeze(v));p===2&&Eg(u)&&(v=ef(v),u=Gg(u,t),u=Hg(u,t,!1),qf(v,u),Ag(m,t,3,v));m=v;t=Ng(e,14);p=c.l;p.h=Math.min(3E5,p.h*2);p.i=Math.min(3E5,p.h+Math.round((Math.random()-.5)*.2*p.h));c.h.setInterval(c.l.getValue());
k===401&&f&&(c.da=f);t&&(c.m+=t);l===void 0&&(l=c.isRetryable(k));l&&(c.j=m.concat(c.j),c.Hb||c.h.enabled||c.h.start());b&&b("net-send-failed",k);++c.P};
c.network&&c.network.send(d,g,h)}}}};
n.xc=function(){ei(this.i,!0);this.flush();ei(this.i,!1)};
n.isRetryable=function(a){return 500<=a&&a<600||a===401||a===0};
function $h(){return"https://play.google.com/log?format=json&hasfast=true"}
function Zh(a,b){this.eb=b=b===void 0?!1:b;this.i=this.locale=null;this.h=new Th;Number.isInteger(a)&&this.h.Qb(a);b||(this.locale=document.documentElement.getAttribute("lang"));ai(this,new Qh)}
Zh.prototype.Qb=function(a){this.h.Qb(a);return this};
function ai(a,b){Lg(a.h,Qh,1,b);Pg(b)||Tg(b,1,1);if(!a.eb){b=fi(a);var c=c===void 0?"":c;var d=xg(b,5);d=d==null||typeof d==="string"?d:void 0;(d!=null?d:c)||Sg(b,5,a.locale)}a.i&&(c=fi(a),Kg(c,Kh,9)||Lg(c,Kh,9,a.i))}
function ci(a,b){Bg(gi(a))&&(a=hi(a),Tg(a,1,b))}
function ei(a,b){Bg(gi(a))&&(a=hi(a),K(a,2,Kf(b)))}
function gi(a){return Kg(a.h,Qh,1)}
function ii(a){var b=b===void 0?Mh:b;var c=a.eb?void 0:window;c?Ph(c,b).then(function(d){a.i=d;d=fi(a);Lg(d,Kh,9,a.i);return!0}).catch(function(){return!1}):Promise.resolve(!1)}
function fi(a){a=gi(a);var b=Kg(a,Dg,11);b||(b=new Dg,Lg(a,Dg,11,b));return b}
function hi(a){a=fi(a);var b=Kg(a,Ih,10);b||(b=new Ih,K(b,2,Kf(!1)),Lg(a,Ih,10,b));return b}
function di(a,b,c,d,e){var f=0,g=0;c=c===void 0?0:c;f=f===void 0?0:f;g=g===void 0?0:g;d=d===void 0?0:d;if(Bg(gi(a))){var h=hi(a);K(h,3,Of(d))}Bg(gi(a))&&(d=hi(a),K(d,4,Of(f)));Bg(gi(a))&&(f=hi(a),K(f,5,Of(g)));a=a.h.clone();g=Date.now().toString();a=K(a,4,Qf(g));b=b.slice();b=Mg(a,Sh,3,b);e&&(a=new Fh,e=K(a,13,Of(e)),a=new Gh,e=Lg(a,Fh,2,e),a=new Rh,e=Lg(a,Gh,1,e),e=Tg(e,2,9),Lg(b,Rh,18,e));c&&K(b,14,Qf(c));return b}
;function ji(){this.Jd=typeof AbortController!=="undefined"}
ji.prototype.send=function(a,b,c){var d=this,e,f,g,h,k,l,m,p,t,r;return A(function(v){switch(v.h){case 1:return f=(e=d.Jd?new AbortController:void 0)?setTimeout(function(){e.abort()},a.timeoutMillis):void 0,Aa(v,2,3),g=Object.assign({},{method:a.requestType,
headers:Object.assign({},a.sd)},a.body&&{body:a.body},a.withCredentials&&{credentials:"include"},{signal:a.timeoutMillis&&e?e.signal:null}),v.yield(fetch(a.url,g),5);case 5:h=v.i;if(h.status!==200){(k=c)==null||k(h.status);v.B(3);break}if((l=b)==null){v.B(7);break}return v.yield(h.text(),8);case 8:l(v.i);case 7:case 3:v.P=[v.j];v.l=0;v.v=0;clearTimeout(f);Ca(v);break;case 2:m=Ba(v);switch((p=m)==null?void 0:p.name){case "AbortError":(t=c)==null||t(408);break;default:(r=c)==null||r(400)}v.B(3)}})};
ji.prototype.ec=function(){return 4};function ki(a,b){G.call(this);this.logSource=a;this.sessionIndex=b;this.j="https://play.google.com/log?format=json&hasfast=true";this.h=null;this.l=!1;this.network=null;this.componentId="";this.pageId=this.i=this.yb=null}
y(ki,G);ki.prototype.Zc=function(){this.m=!0;return this};
function li(a){a.network||(a.network=new ji);var b=new Yh({logSource:a.logSource,Jb:a.Jb?a.Jb:Ch,sessionIndex:a.sessionIndex,lf:a.j,eb:a.l,Hb:!1,Zc:a.m,Tc:a.Tc,network:a.network});Ec(a,b);if(a.h){var c=a.h,d=fi(b.i);Sg(d,7,c)}a.componentId&&(b.componentId=a.componentId);a.yb&&(b.yb=a.yb);a.pageId&&(b.pageId=a.pageId);a.i&&((d=a.i)?(b.experimentIds||(b.experimentIds=new Dh),c=b.experimentIds,d=d.serialize(),Sg(c,4,d)):b.experimentIds&&K(b.experimentIds,4));ii(b.i);a.network.Qb&&a.network.Qb(a.logSource);
a.network.We&&a.network.We(b);return b}
;function mi(a,b,c,d,e,f,g){a=a===void 0?-1:a;b=b===void 0?"":b;c=c===void 0?"":c;d=d===void 0?!1:d;e=e===void 0?"":e;G.call(this);this.logSource=a;this.componentId=b;f?b=f:(a=new ki(a,"0"),a.componentId=b,Ec(this,a),c!==""&&(a.j=c),d&&(a.l=!0),e&&(a.h=e),g&&(a.network=g),b=li(a));this.h=b}
y(mi,G);
mi.prototype.flush=function(a){var b=a||[];if(b.length){a=new qh;for(var c=[],d=0;d<b.length;d++){var e=b[d];var f=new ph;f=Sg(f,1,e.l);for(var g=[],h=0;h<e.fields.length;h++)g.push(e.fields[h].F);f=Fg(f,g,Vf);g=[];h=[];for(var k=x(e.h.keys()),l=k.next();!l.done;l=k.next())h.push(l.value.split(","));for(k=0;k<h.length;k++){l=h[k];var m=e.j;for(var p=e.yc(l)||[],t=[],r=0;r<p.length;r++){var v=p[r],u=v&&v.h;v=new mh;switch(m){case 3:u=Number(u);Number.isFinite(u)&&Ig(v,1,nh,Qf(u));break;case 2:u=Number(u);
if(u!=null&&typeof u!=="number")throw Error("Value of float/double field must be a number, found "+typeof u+": "+u);Ig(v,2,nh,u)}t.push(v)}m=t;for(p=0;p<m.length;p++){t=m[p];r=new oh;t=Lg(r,mh,2,t);r=l;v=[];u=[];for(var z=0;z<e.fields.length;z++)u.push(e.fields[z].G);for(z=0;z<u.length;z++){var F=u[z],J=r[z],O=new kh;switch(F){case 3:Ig(O,1,lh,Wf(String(J)));break;case 2:F=Number(J);Number.isFinite(F)&&Ig(O,2,lh,Of(F));break;case 1:Ig(O,3,lh,Kf(J==="true"))}v.push(O)}Mg(t,kh,1,v);g.push(t)}}Mg(f,
oh,4,g);c.push(f);e.clear()}Mg(a,ph,1,c);b=this.h;b.H&&(a instanceof Sh?b.log(a):(c=new Sh,a=a.serialize(),a=Sg(c,8,a),b.log(a)));this.h.flush()}};function ni(){}
ni.prototype.serialize=function(a){var b=[];oi(this,a,b);return b.join("")};
function oi(a,b,c){if(b==null)c.push("null");else{if(typeof b=="object"){if(Array.isArray(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),oi(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],typeof f!="function"&&(c.push(e),pi(d,c),c.push(":"),oi(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":pi(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var qi={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\v":"\\u000b"},ri=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function pi(a,b){b.push('"',a.replace(ri,function(c){var d=qi[c];d||(d="\\u"+(c.charCodeAt(0)|65536).toString(16).slice(1),qi[c]=d);return d}),'"')}
;function si(){}
si.prototype.h=null;si.prototype.getOptions=function(){var a;(a=this.h)||(a=this.h={});return a};var ti;function ui(){}
$a(ui,si);ti=new ui;function vi(a){wd.call(this);this.headers=new Map;this.Ga=a||null;this.i=!1;this.P=this.T=null;this.l=this.da="";this.j=this.ba=this.m=this.W=!1;this.H=0;this.A=null;this.xa="";this.ia=!1}
$a(vi,wd);var wi=/^https?$/i,xi=["POST","PUT"],yi=[];function zi(a,b,c,d,e,f,g){var h=new vi;yi.push(h);b&&h.listen("complete",b);h.h.add("ready",h.Qd,!0,void 0,void 0);f&&(h.H=Math.max(0,f));g&&(h.ia=g);h.send(a,c,d,e)}
n=vi.prototype;n.Qd=function(){this.dispose();Ib(yi,this)};
n.send=function(a,b,c,d){if(this.T)throw Error("[goog.net.XhrIo] Object is active with another request="+this.da+"; newUri="+a);b=b?b.toUpperCase():"GET";this.da=a;this.l="";this.W=!1;this.i=!0;this.T=new XMLHttpRequest;this.P=this.Ga?this.Ga.getOptions():ti.getOptions();this.T.onreadystatechange=Xa(this.md,this);try{this.getStatus(),this.ba=!0,this.T.open(b,String(a),!0),this.ba=!1}catch(g){this.getStatus();Ai(this,g);return}a=c||"";c=new Map(this.headers);if(d)if(Object.getPrototypeOf(d)===Object.prototype)for(var e in d)c.set(e,
d[e]);else if(typeof d.keys==="function"&&typeof d.get==="function"){e=x(d.keys());for(var f=e.next();!f.done;f=e.next())f=f.value,c.set(f,d.get(f))}else throw Error("Unknown input type for opt_headers: "+String(d));d=Array.from(c.keys()).find(function(g){return"content-type"==g.toLowerCase()});
e=C.FormData&&a instanceof C.FormData;!(Cb(xi,b)>=0)||d||e||c.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");b=x(c);for(d=b.next();!d.done;d=b.next())c=x(d.value),d=c.next().value,c=c.next().value,this.T.setRequestHeader(d,c);this.xa&&(this.T.responseType=this.xa);"withCredentials"in this.T&&this.T.withCredentials!==this.ia&&(this.T.withCredentials=this.ia);try{Bi(this),this.H>0&&(this.getStatus(),this.A=le(this.hf,this.H,this)),this.getStatus(),this.m=!0,this.T.send(a),this.m=
!1}catch(g){this.getStatus(),Ai(this,g)}};
n.hf=function(){typeof Na!="undefined"&&this.T&&(this.l="Timed out after "+this.H+"ms, aborting",this.getStatus(),xd(this,"timeout"),this.abort(8))};
function Ai(a,b){a.i=!1;a.T&&(a.j=!0,a.T.abort(),a.j=!1);a.l=b;Ci(a);Di(a)}
function Ci(a){a.W||(a.W=!0,xd(a,"complete"),xd(a,"error"))}
n.abort=function(){this.T&&this.i&&(this.getStatus(),this.i=!1,this.j=!0,this.T.abort(),this.j=!1,xd(this,"complete"),xd(this,"abort"),Di(this))};
n.U=function(){this.T&&(this.i&&(this.i=!1,this.j=!0,this.T.abort(),this.j=!1),Di(this,!0));vi.Ba.U.call(this)};
n.md=function(){this.V||(this.ba||this.m||this.j?Ei(this):this.ze())};
n.ze=function(){Ei(this)};
function Ei(a){if(a.i&&typeof Na!="undefined")if(a.P[1]&&Fi(a)==4&&a.getStatus()==2)a.getStatus();else if(a.m&&Fi(a)==4)le(a.md,0,a);else if(xd(a,"readystatechange"),a.isComplete()){a.getStatus();a.i=!1;try{if(Gi(a))xd(a,"complete"),xd(a,"success");else{try{var b=Fi(a)>2?a.T.statusText:""}catch(c){b=""}a.l=b+" ["+a.getStatus()+"]";Ci(a)}}finally{Di(a)}}}
function Di(a,b){if(a.T){Bi(a);var c=a.T,d=a.P[0]?function(){}:null;
a.T=null;a.P=null;b||xd(a,"ready");try{c.onreadystatechange=d}catch(e){}}}
function Bi(a){a.A&&(C.clearTimeout(a.A),a.A=null)}
n.isActive=function(){return!!this.T};
n.isComplete=function(){return Fi(this)==4};
function Gi(a){var b=a.getStatus();a:switch(b){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var c=!0;break a;default:c=!1}if(!c){if(b=b===0)a=nc(1,String(a.da)),!a&&C.self&&C.self.location&&(a=C.self.location.protocol.slice(0,-1)),b=!wi.test(a?a.toLowerCase():"");c=b}return c}
function Fi(a){return a.T?a.T.readyState:0}
n.getStatus=function(){try{return Fi(this)>2?this.T.status:-1}catch(a){return-1}};
n.getLastError=function(){return typeof this.l==="string"?this.l:String(this.l)};function Hi(){}
Hi.prototype.send=function(a,b,c){b=b===void 0?function(){}:b;
c=c===void 0?function(){}:c;
zi(a.url,function(d){d=d.target;if(Gi(d)){try{var e=d.T?d.T.responseText:""}catch(f){e=""}b(e)}else c(d.getStatus())},a.requestType,a.body,a.sd,a.timeoutMillis,a.withCredentials)};
Hi.prototype.ec=function(){return 1};function Ii(a,b,c){this.logger=a;this.event=b;if(c===void 0||c)this.h=Ji()}
Ii.prototype.start=function(){this.h=Ji()};
Ii.prototype.done=function(){this.h!=null&&this.logger.jc(this.event,Ji()-this.h)};
function Ki(){}
Ki.prototype.Cc=function(){};
Ki.prototype.jc=function(){};
Ki.prototype.Ha=function(){};
Ki.prototype.Aa=function(){};
function Li(a,b,c,d,e){d=d===void 0?"":d;G.call(this);this.Ea=b;this.A=d;this.i=new Map;this.j=new Map;b=new ki(1828,"0");b.h="21";b.network=new Hi;e&&(d=new fh,e=Fg(d,e,Nf),b.i=e);this.m=new mi(1828,"","",!1,"",li(b));this.h=new me(this.m);c&&(this.h.l=1E5,c=this.h,c.flushInterval=3E4,c.h.setInterval(3E4));this.da=new se(this.h);this.ga=new ve(this.h);this.ia=new we(this.h);this.ba=new re(this.h);this.H=new te(this.h);this.P=new ue(this.h);this.errorCount=new ze(this.h);this.W=new ye(this.h);new xe(this.h);
new Ae(this.h);new Be(this.h);new Ce(this.h);this.l=hh(a);a=new qe(this.h);this.i.set("h",1);this.i.set("u",2);this.i.set("k",3);this.i.set("P",4);this.i.set("p",5);this.j.set(25,1);this.j.set(26,2);this.j.set(27,3);this.j.set(28,4);a.h.Ab("/client_streamz/bg/fic",this.Ea);Ec(this,this.m);Ec(this,this.h)}
y(Li,G);Li.prototype.Cc=function(){this.ga.h.Ab("/client_streamz/bg/fsc",this.l,this.Ea)};
Li.prototype.jc=function(a,b){if(a==="t")this.da.record(b,this.l,this.Ea);else if(a==="n")this.ia.record(b,this.l,this.Ea);else if(a==="h"||a==="u"||a==="k"||a==="P"||a==="p"){if(a=this.i.get(a))this.H.h.Ab("/client_streamz/bg/fcc",a,this.Ea),this.P.record(b,a,this.Ea)}else this.W.record(b,a,"",this.A,this.Ea)};
Li.prototype.Ha=function(a){var b=this.j.get(a);b?this.ba.h.Ab("/client_streamz/bg/fiec",this.l,this.Ea,b):this.errorCount.h.Ab("/client_streamz/bg/cec",a,"",this.A,this.Ea)};
Li.prototype.Aa=function(){this.h.Aa()};
function Ji(){var a,b,c;return(c=(a=globalThis.performance)==null?void 0:(b=a.now)==null?void 0:b.call(a))!=null?c:Date.now()}
;function Mi(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function Ni(a){function b(r,v,u){Promise.resolve().then(function(){p.done();d.h&&d.ea.Aa();m.resolve({Md:r,Ze:v,Rg:u})})}
function c(r,v,u,z){if(h){var F="k";v?F="h":u&&(F="u");F!=="k"?z!==0&&d.ea.jc(F,r):d.i<=0?(d.ea.jc(F,r),d.i=Math.floor(Math.random()*200)):d.i--}}
G.call(this);var d=this;this.h=!1;this.i=Math.floor(Math.random()*200);var e=a.program;var f=a.ee;var g=Math.random(),h=g<.3;a.Od!=null&&(h=g<a.Od);h&&(this.h=!0);var k=new G;this.addOnDisposeCallback(function(){d.j.then(function(r){r=r.Ze;d.ea.Aa();r==null||r();k.dispose()})});
if(a.Fe!==!1)if(a.ea)this.ea=a.ea;else{var l;Ec(k,this.ea=new Li(f,(l=a.Ea)!=null?l:"_",this.h))}else this.ea=new Ki;var m=new Mi;this.j=m.promise;var p=new Ii(this.ea,"t",!1);if(!C[f])throw this.ea.Ha(25),this.ea.Aa(),Error("EGOU");if(!C[f].a)throw this.ea.Ha(26),this.ea.Aa(),Error("ELIU");try{var t=C[f].a;p.start();this.l=x(t(e,b,!0,a.bh,c)).next().value;this.Ye=m.promise.then(function(){})}catch(r){throw this.ea.Ha(28),this.ea.Aa(),r;
}}
y(Ni,G);Ni.prototype.snapshot=function(a){var b=this;if(this.V)throw Error("Already disposed");this.ea.Cc();return this.j.then(function(c){var d=c.Md;return new Promise(function(e){var f=new Ii(b.ea,"n");d(function(g){f.done();b.h&&b.ea.Aa();e(g)},[a.Yc,
a.af,a.nf,a.bf])})})};
Ni.prototype.yd=function(a){if(this.V)throw Error("Already disposed");this.ea.Cc();var b=new Ii(this.ea,"n");a=this.l([a.Yc,a.af,a.nf,a.bf]);b.done();this.h&&this.ea.Aa();return a};var Oi=window;ib("csi.gstatic.com");ib("googleads.g.doubleclick.net");ib("partner.googleadservices.com");ib("pubads.g.doubleclick.net");ib("securepubads.g.doubleclick.net");ib("tpc.googlesyndication.com");function Pi(a){var b=Qi;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function Ri(){var a=[];Pi(function(b){a.push(b)});
return a}
var Qi={pf:"allow-forms",qf:"allow-modals",rf:"allow-orientation-lock",sf:"allow-pointer-lock",tf:"allow-popups",uf:"allow-popups-to-escape-sandbox",vf:"allow-presentation",wf:"allow-same-origin",xf:"allow-scripts",yf:"allow-top-navigation",zf:"allow-top-navigation-by-user-activation"},Si=Cd(function(){return Ri()});
function Ti(){var a=Ui(),b={};Db(Si(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function Ui(){var a=a===void 0?document:a;return a.createElement("iframe")}
;function Vi(a){typeof a=="number"&&(a=Math.round(a)+"px");return a}
;var Wi=(new Date).getTime();function Xi(){var a=Yi;return eh(function(b){for(var c in a)if(b===a[c]&&!/^[0-9]+$/.test(c))return!0;return!1})}
;function Zi(a){wd.call(this);var b=this;this.A=this.j=0;this.Da=a!=null?a:{pa:function(e,f){return setTimeout(e,f)},
qa:function(e){clearTimeout(e)}};
var c,d;this.i=(d=(c=window.navigator)==null?void 0:c.onLine)!=null?d:!0;this.l=function(){return A(function(e){return e.yield($i(b),0)})};
window.addEventListener("offline",this.l);window.addEventListener("online",this.l);this.A||aj(this)}
y(Zi,wd);function bj(){var a=cj;Zi.h||(Zi.h=new Zi(a));return Zi.h}
Zi.prototype.dispose=function(){window.removeEventListener("offline",this.l);window.removeEventListener("online",this.l);this.Da.qa(this.A);delete Zi.h};
Zi.prototype.va=function(){return this.i};
function aj(a){a.A=a.Da.pa(function(){var b;return A(function(c){if(c.h==1)return a.i?((b=window.navigator)==null?0:b.onLine)?c.B(3):c.yield($i(a),3):c.yield($i(a),3);aj(a);c.h=0})},3E4)}
function $i(a,b){return a.m?a.m:a.m=new Promise(function(c){var d,e,f,g;return A(function(h){switch(h.h){case 1:return d=window.AbortController?new window.AbortController:void 0,f=(e=d)==null?void 0:e.signal,g=!1,Aa(h,2,3),d&&(a.j=a.Da.pa(function(){d.abort()},b||2E4)),h.yield(fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:h.P=[h.j];h.l=0;h.v=0;a.m=void 0;a.j&&(a.Da.qa(a.j),a.j=0);g!==a.i&&(a.i=g,a.i?xd(a,"networkstatus-online"):xd(a,"networkstatus-offline"));c(g);Ca(h);break;case 2:Ba(h),g=!1,h.B(3)}})})}
;function dj(){this.data=[];this.h=-1}
dj.prototype.set=function(a,b){b=b===void 0?!0:b;0<=a&&a<52&&Number.isInteger(a)&&this.data[a]!==b&&(this.data[a]=b,this.h=-1)};
dj.prototype.get=function(a){return!!this.data[a]};
function ej(a){a.h===-1&&(a.h=a.data.reduce(function(b,c,d){return b+(c?Math.pow(2,d):0)},0));
return a.h}
;function fj(){this.blockSize=-1}
;function gj(){this.blockSize=-1;this.blockSize=64;this.h=[];this.v=[];this.m=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.l=this.i=0;this.reset()}
$a(gj,fj);gj.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.l=this.i=0};
function hj(a,b,c){c||(c=0);var d=a.m;if(typeof b==="string")for(var e=0;e<16;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;e<16;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;e<80;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.h[0];c=a.h[1];var g=a.h[2],h=a.h[3],k=a.h[4];for(e=0;e<80;e++){if(e<40)if(e<20){f=h^c&(g^h);var l=1518500249}else f=c^g^h,l=1859775393;else e<60?(f=c&g|h&(c|g),l=2400959708):
(f=c^g^h,l=3395469782);f=(b<<5|b>>>27)+f+k+l+d[e]&4294967295;k=h;h=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+g&4294967295;a.h[3]=a.h[3]+h&4294967295;a.h[4]=a.h[4]+k&4294967295}
gj.prototype.update=function(a,b){if(a!=null){b===void 0&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.v,f=this.i;d<b;){if(f==0)for(;d<=c;)hj(this,a,d),d+=this.blockSize;if(typeof a==="string")for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){hj(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){hj(this,e);f=0;break}}this.i=f;this.l+=b}};
gj.prototype.digest=function(){var a=[],b=this.l*8;this.i<56?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;c>=56;c--)this.v[c]=b&255,b/=256;hj(this,this.v);for(c=b=0;c<5;c++)for(var d=24;d>=0;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function ij(a){return typeof a.className=="string"?a.className:a.getAttribute&&a.getAttribute("class")||""}
function jj(a,b){typeof a.className=="string"?a.className=b:a.setAttribute&&a.setAttribute("class",b)}
function kj(a,b){a.classList?b=a.classList.contains(b):(a=a.classList?a.classList:ij(a).match(/\S+/g)||[],b=Cb(a,b)>=0);return b}
function lj(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):kj(a,"inverted-hdpi")&&jj(a,Array.prototype.filter.call(a.classList?a.classList:ij(a).match(/\S+/g)||[],function(b){return b!="inverted-hdpi"}).join(" "))}
;function mj(){}
mj.prototype.next=function(){return nj};
var nj={done:!0,value:void 0};mj.prototype.mb=function(){return this};function oj(a){if(a instanceof pj||a instanceof qj||a instanceof rj)return a;if(typeof a.next=="function")return new pj(function(){return a});
if(typeof a[Symbol.iterator]=="function")return new pj(function(){return a[Symbol.iterator]()});
if(typeof a.mb=="function")return new pj(function(){return a.mb()});
throw Error("Not an iterator or iterable.");}
function pj(a){this.h=a}
pj.prototype.mb=function(){return new qj(this.h())};
pj.prototype[Symbol.iterator]=function(){return new rj(this.h())};
pj.prototype.i=function(){return new rj(this.h())};
function qj(a){this.h=a}
y(qj,mj);qj.prototype.next=function(){return this.h.next()};
qj.prototype[Symbol.iterator]=function(){return new rj(this.h)};
qj.prototype.i=function(){return new rj(this.h)};
function rj(a){pj.call(this,function(){return a});
this.j=a}
y(rj,pj);rj.prototype.next=function(){return this.j.next()};function M(a){G.call(this);this.m=1;this.j=[];this.l=0;this.h=[];this.i={};this.A=!!a}
$a(M,G);n=M.prototype;n.subscribe=function(a,b,c){var d=this.i[a];d||(d=this.i[a]=[]);var e=this.m;this.h[e]=a;this.h[e+1]=b;this.h[e+2]=c;this.m=e+3;d.push(e);return e};
n.unsubscribe=function(a,b,c){if(a=this.i[a]){var d=this.h;if(a=a.find(function(e){return d[e+1]==b&&d[e+2]==c}))return this.Bb(a)}return!1};
n.Bb=function(a){var b=this.h[a];if(b){var c=this.i[b];this.l!=0?(this.j.push(a),this.h[a+1]=function(){}):(c&&Ib(c,a),delete this.h[a],delete this.h[a+1],delete this.h[a+2])}return!!b};
n.Ya=function(a,b){var c=this.i[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.A)for(e=0;e<c.length;e++){var g=c[e];sj(this.h[g+1],this.h[g+2],d)}else{this.l++;try{for(e=0,f=c.length;e<f&&!this.V;e++)g=c[e],this.h[g+1].apply(this.h[g+2],d)}finally{if(this.l--,this.j.length>0&&this.l==0)for(;c=this.j.pop();)this.Bb(c)}}return e!=0}return!1};
function sj(a,b,c){Rd(function(){a.apply(b,c)})}
n.clear=function(a){if(a){var b=this.i[a];b&&(b.forEach(this.Bb,this),delete this.i[a])}else this.h.length=0,this.i={}};
n.U=function(){M.Ba.U.call(this);this.clear();this.j.length=0};function tj(a){this.h=a}
tj.prototype.set=function(a,b){b===void 0?this.h.remove(a):this.h.set(a,(new ni).serialize(b))};
tj.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(b!==null)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
tj.prototype.remove=function(a){this.h.remove(a)};function uj(a){this.h=a}
$a(uj,tj);function vj(a){this.data=a}
function wj(a){return a===void 0||a instanceof vj?a:new vj(a)}
uj.prototype.set=function(a,b){uj.Ba.set.call(this,a,wj(b))};
uj.prototype.i=function(a){a=uj.Ba.get.call(this,a);if(a===void 0||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
uj.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,a===void 0)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function xj(a){this.h=a}
$a(xj,uj);xj.prototype.set=function(a,b,c){if(b=wj(b)){if(c){if(c<Za()){xj.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Za()}xj.Ba.set.call(this,a,b)};
xj.prototype.i=function(a){var b=xj.Ba.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Za()||c&&c>Za())xj.prototype.remove.call(this,a);else return b}};function yj(){}
;function zj(){}
$a(zj,yj);zj.prototype[Symbol.iterator]=function(){return oj(this.mb(!0)).i()};
zj.prototype.clear=function(){var a=Array.from(this);a=x(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function Aj(a){this.h=a;this.i=null}
$a(Aj,zj);n=Aj.prototype;n.isAvailable=function(){var a=this.h;if(a)try{a.setItem("__sak","1");a.removeItem("__sak");var b=!0}catch(c){b=c instanceof DOMException&&(c.name==="QuotaExceededError"||c.code===22||c.code===1014||c.name==="NS_ERROR_DOM_QUOTA_REACHED")&&a&&a.length!==0}else b=!1;return this.i=b};
n.set=function(a,b){Bj(this);try{this.h.setItem(a,b)}catch(c){if(this.h.length==0)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
n.get=function(a){Bj(this);a=this.h.getItem(a);if(typeof a!=="string"&&a!==null)throw"Storage mechanism: Invalid value was encountered";return a};
n.remove=function(a){Bj(this);this.h.removeItem(a)};
n.mb=function(a){Bj(this);var b=0,c=this.h,d=new mj;d.next=function(){if(b>=c.length)return nj;var e=c.key(b++);if(a)return{value:e,done:!1};e=c.getItem(e);if(typeof e!=="string")throw"Storage mechanism: Invalid value was encountered";return{value:e,done:!1}};
return d};
n.clear=function(){Bj(this);this.h.clear()};
n.key=function(a){Bj(this);return this.h.key(a)};
function Bj(a){if(a.h==null)throw Error("Storage mechanism: Storage unavailable");var b;((b=a.i)!=null?b:a.isAvailable())||Kd(Error("Storage mechanism: Storage unavailable"))}
;function Cj(){var a=null;try{a=C.localStorage||null}catch(b){}Aj.call(this,a)}
$a(Cj,Aj);function Dj(a,b){this.i=a;this.h=b+"::"}
$a(Dj,zj);Dj.prototype.set=function(a,b){this.i.set(this.h+a,b)};
Dj.prototype.get=function(a){return this.i.get(this.h+a)};
Dj.prototype.remove=function(a){this.i.remove(this.h+a)};
Dj.prototype.mb=function(a){var b=this.i[Symbol.iterator](),c=this,d=new mj;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.h.length)!=c.h;){e=b.next();if(e.done)return e;e=e.value}return{value:a?e.slice(c.h.length):c.i.get(e),done:!1}};
return d};/*

 (The MIT License)

 Copyright (C) 2014 by Vitaly Puzrin

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 -----------------------------------------------------------------------------
 Ported from zlib, which is under the following license
 https://github.com/madler/zlib/blob/master/zlib.h

 zlib.h -- interface of the 'zlib' general purpose compression library
   version 1.2.8, April 28th, 2013
   Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu
   The data format used by the zlib library is described by RFCs (Request for
   Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
   (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/
var N={},Ej=typeof Uint8Array!=="undefined"&&typeof Uint16Array!=="undefined"&&typeof Int32Array!=="undefined";N.assign=function(a){for(var b=Array.prototype.slice.call(arguments,1);b.length;){var c=b.shift();if(c){if(typeof c!=="object")throw new TypeError(c+"must be non-object");for(var d in c)Object.prototype.hasOwnProperty.call(c,d)&&(a[d]=c[d])}}return a};
N.Nc=function(a,b){if(a.length===b)return a;if(a.subarray)return a.subarray(0,b);a.length=b;return a};
var Fj={nb:function(a,b,c,d,e){if(b.subarray&&a.subarray)a.set(b.subarray(c,c+d),e);else for(var f=0;f<d;f++)a[e+f]=b[c+f]},
cd:function(a){var b,c;var d=c=0;for(b=a.length;d<b;d++)c+=a[d].length;var e=new Uint8Array(c);d=c=0;for(b=a.length;d<b;d++){var f=a[d];e.set(f,c);c+=f.length}return e}},Gj={nb:function(a,b,c,d,e){for(var f=0;f<d;f++)a[e+f]=b[c+f]},
cd:function(a){return[].concat.apply([],a)}};
N.Xe=function(){Ej?(N.lb=Uint8Array,N.Ja=Uint16Array,N.Hd=Int32Array,N.assign(N,Fj)):(N.lb=Array,N.Ja=Array,N.Hd=Array,N.assign(N,Gj))};
N.Xe();var Hj=!0;try{new Uint8Array(1)}catch(a){Hj=!1}
function Ij(a){var b,c,d=a.length,e=0;for(b=0;b<d;b++){var f=a.charCodeAt(b);if((f&64512)===55296&&b+1<d){var g=a.charCodeAt(b+1);(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)}e+=f<128?1:f<2048?2:f<65536?3:4}var h=new N.lb(e);for(b=c=0;c<e;b++)f=a.charCodeAt(b),(f&64512)===55296&&b+1<d&&(g=a.charCodeAt(b+1),(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)),f<128?h[c++]=f:(f<2048?h[c++]=192|f>>>6:(f<65536?h[c++]=224|f>>>12:(h[c++]=240|f>>>18,h[c++]=128|f>>>12&63),h[c++]=128|f>>>
6&63),h[c++]=128|f&63);return h}
;var Jj={};Jj=function(a,b,c,d){var e=a&65535|0;a=a>>>16&65535|0;for(var f;c!==0;){f=c>2E3?2E3:c;c-=f;do e=e+b[d++]|0,a=a+e|0;while(--f);e%=65521;a%=65521}return e|a<<16|0};for(var Kj={},Lj,Mj=[],Nj=0;Nj<256;Nj++){Lj=Nj;for(var Oj=0;Oj<8;Oj++)Lj=Lj&1?3988292384^Lj>>>1:Lj>>>1;Mj[Nj]=Lj}Kj=function(a,b,c,d){c=d+c;for(a^=-1;d<c;d++)a=a>>>8^Mj[(a^b[d])&255];return a^-1};var Pj={};Pj={2:"need dictionary",1:"stream end",0:"","-1":"file error","-2":"stream error","-3":"data error","-4":"insufficient memory","-5":"buffer error","-6":"incompatible version"};function Qj(a){for(var b=a.length;--b>=0;)a[b]=0}
var Rj=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0],Sj=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],Tj=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7],Uj=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15],Vj=Array(576);Qj(Vj);var Wj=Array(60);Qj(Wj);var Xj=Array(512);Qj(Xj);var Yj=Array(256);Qj(Yj);var Zj=Array(29);Qj(Zj);var ak=Array(30);Qj(ak);function bk(a,b,c,d,e){this.zd=a;this.Zd=b;this.Yd=c;this.Td=d;this.we=e;this.gd=a&&a.length}
var ck,dk,ek;function fk(a,b){this.bd=a;this.vb=0;this.Wa=b}
function gk(a,b){a.Z[a.pending++]=b&255;a.Z[a.pending++]=b>>>8&255}
function hk(a,b,c){a.ja>16-c?(a.oa|=b<<a.ja&65535,gk(a,a.oa),a.oa=b>>16-a.ja,a.ja+=c-16):(a.oa|=b<<a.ja&65535,a.ja+=c)}
function ik(a,b,c){hk(a,c[b*2],c[b*2+1])}
function jk(a,b){var c=0;do c|=a&1,a>>>=1,c<<=1;while(--b>0);return c>>>1}
function kk(a,b,c){var d=Array(16),e=0,f;for(f=1;f<=15;f++)d[f]=e=e+c[f-1]<<1;for(c=0;c<=b;c++)e=a[c*2+1],e!==0&&(a[c*2]=jk(d[e]++,e))}
function lk(a){var b;for(b=0;b<286;b++)a.ra[b*2]=0;for(b=0;b<30;b++)a.bb[b*2]=0;for(b=0;b<19;b++)a.ka[b*2]=0;a.ra[512]=1;a.Pa=a.zb=0;a.ya=a.matches=0}
function mk(a){a.ja>8?gk(a,a.oa):a.ja>0&&(a.Z[a.pending++]=a.oa);a.oa=0;a.ja=0}
function nk(a,b,c){mk(a);gk(a,c);gk(a,~c);N.nb(a.Z,a.window,b,c,a.pending);a.pending+=c}
function ok(a,b,c,d){var e=b*2,f=c*2;return a[e]<a[f]||a[e]===a[f]&&d[b]<=d[c]}
function pk(a,b,c){for(var d=a.aa[c],e=c<<1;e<=a.Na;){e<a.Na&&ok(b,a.aa[e+1],a.aa[e],a.depth)&&e++;if(ok(b,d,a.aa[e],a.depth))break;a.aa[c]=a.aa[e];c=e;e<<=1}a.aa[c]=d}
function qk(a,b,c){var d=0;if(a.ya!==0){do{var e=a.Z[a.Gb+d*2]<<8|a.Z[a.Gb+d*2+1];var f=a.Z[a.Bc+d];d++;if(e===0)ik(a,f,b);else{var g=Yj[f];ik(a,g+256+1,b);var h=Rj[g];h!==0&&(f-=Zj[g],hk(a,f,h));e--;g=e<256?Xj[e]:Xj[256+(e>>>7)];ik(a,g,c);h=Sj[g];h!==0&&(e-=ak[g],hk(a,e,h))}}while(d<a.ya)}ik(a,256,b)}
function rk(a,b){var c=b.bd,d=b.Wa.zd,e=b.Wa.gd,f=b.Wa.Td,g,h=-1;a.Na=0;a.qb=573;for(g=0;g<f;g++)c[g*2]!==0?(a.aa[++a.Na]=h=g,a.depth[g]=0):c[g*2+1]=0;for(;a.Na<2;){var k=a.aa[++a.Na]=h<2?++h:0;c[k*2]=1;a.depth[k]=0;a.Pa--;e&&(a.zb-=d[k*2+1])}b.vb=h;for(g=a.Na>>1;g>=1;g--)pk(a,c,g);k=f;do g=a.aa[1],a.aa[1]=a.aa[a.Na--],pk(a,c,1),d=a.aa[1],a.aa[--a.qb]=g,a.aa[--a.qb]=d,c[k*2]=c[g*2]+c[d*2],a.depth[k]=(a.depth[g]>=a.depth[d]?a.depth[g]:a.depth[d])+1,c[g*2+1]=c[d*2+1]=k,a.aa[1]=k++,pk(a,c,1);while(a.Na>=
2);a.aa[--a.qb]=a.aa[1];g=b.bd;k=b.vb;d=b.Wa.zd;e=b.Wa.gd;f=b.Wa.Zd;var l=b.Wa.Yd,m=b.Wa.we,p,t=0;for(p=0;p<=15;p++)a.Ka[p]=0;g[a.aa[a.qb]*2+1]=0;for(b=a.qb+1;b<573;b++){var r=a.aa[b];p=g[g[r*2+1]*2+1]+1;p>m&&(p=m,t++);g[r*2+1]=p;if(!(r>k)){a.Ka[p]++;var v=0;r>=l&&(v=f[r-l]);var u=g[r*2];a.Pa+=u*(p+v);e&&(a.zb+=u*(d[r*2+1]+v))}}if(t!==0){do{for(p=m-1;a.Ka[p]===0;)p--;a.Ka[p]--;a.Ka[p+1]+=2;a.Ka[m]--;t-=2}while(t>0);for(p=m;p!==0;p--)for(r=a.Ka[p];r!==0;)d=a.aa[--b],d>k||(g[d*2+1]!==p&&(a.Pa+=(p-g[d*
2+1])*g[d*2],g[d*2+1]=p),r--)}kk(c,h,a.Ka)}
function sk(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);b[(c+1)*2+1]=65535;for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];++g<h&&l===f||(g<k?a.ka[l*2]+=g:l!==0?(l!==e&&a.ka[l*2]++,a.ka[32]++):g<=10?a.ka[34]++:a.ka[36]++,g=0,e=l,f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4))}}
function tk(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];if(!(++g<h&&l===f)){if(g<k){do ik(a,l,a.ka);while(--g!==0)}else l!==0?(l!==e&&(ik(a,l,a.ka),g--),ik(a,16,a.ka),hk(a,g-3,2)):g<=10?(ik(a,17,a.ka),hk(a,g-3,3)):(ik(a,18,a.ka),hk(a,g-11,7));g=0;e=l;f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4)}}}
function uk(a){var b=4093624447,c;for(c=0;c<=31;c++,b>>>=1)if(b&1&&a.ra[c*2]!==0)return 0;if(a.ra[18]!==0||a.ra[20]!==0||a.ra[26]!==0)return 1;for(c=32;c<256;c++)if(a.ra[c*2]!==0)return 1;return 0}
var vk=!1;function wk(a,b,c){a.Z[a.Gb+a.ya*2]=b>>>8&255;a.Z[a.Gb+a.ya*2+1]=b&255;a.Z[a.Bc+a.ya]=c&255;a.ya++;b===0?a.ra[c*2]++:(a.matches++,b--,a.ra[(Yj[c]+256+1)*2]++,a.bb[(b<256?Xj[b]:Xj[256+(b>>>7)])*2]++);return a.ya===a.Mb-1}
;function xk(a,b){a.msg=Pj[b];return b}
function yk(a){for(var b=a.length;--b>=0;)a[b]=0}
function zk(a){var b=a.state,c=b.pending;c>a.R&&(c=a.R);c!==0&&(N.nb(a.output,b.Z,b.Ob,c,a.wb),a.wb+=c,b.Ob+=c,a.Oc+=c,a.R-=c,b.pending-=c,b.pending===0&&(b.Ob=0))}
function Ak(a,b){var c=a.ta>=0?a.ta:-1,d=a.o-a.ta,e=0;if(a.level>0){a.K.wc===2&&(a.K.wc=uk(a));rk(a,a.ic);rk(a,a.cc);sk(a,a.ra,a.ic.vb);sk(a,a.bb,a.cc.vb);rk(a,a.Uc);for(e=18;e>=3&&a.ka[Uj[e]*2+1]===0;e--);a.Pa+=3*(e+1)+14;var f=a.Pa+3+7>>>3;var g=a.zb+3+7>>>3;g<=f&&(f=g)}else f=g=d+5;if(d+4<=f&&c!==-1)hk(a,b?1:0,3),nk(a,c,d);else if(a.strategy===4||g===f)hk(a,2+(b?1:0),3),qk(a,Vj,Wj);else{hk(a,4+(b?1:0),3);c=a.ic.vb+1;d=a.cc.vb+1;e+=1;hk(a,c-257,5);hk(a,d-1,5);hk(a,e-4,4);for(f=0;f<e;f++)hk(a,a.ka[Uj[f]*
2+1],3);tk(a,a.ra,c-1);tk(a,a.bb,d-1);qk(a,a.ra,a.bb)}lk(a);b&&mk(a);a.ta=a.o;zk(a.K)}
function R(a,b){a.Z[a.pending++]=b}
function Bk(a,b){a.Z[a.pending++]=b>>>8&255;a.Z[a.pending++]=b&255}
function Ck(a,b){var c=a.kd,d=a.o,e=a.wa,f=a.ld,g=a.o>a.ma-262?a.o-(a.ma-262):0,h=a.window,k=a.Xa,l=a.Ia,m=a.o+258,p=h[d+e-1],t=h[d+e];a.wa>=a.ed&&(c>>=2);f>a.u&&(f=a.u);do{var r=b;if(h[r+e]===t&&h[r+e-1]===p&&h[r]===h[d]&&h[++r]===h[d+1]){d+=2;for(r++;h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&h[++d]===h[++r]&&d<m;);r=258-(m-d);d=m-258;if(r>e){a.ub=b;e=r;if(r>=f)break;p=h[d+e-1];t=h[d+e]}}}while((b=l[b&k])>g&&--c!==0);return e<=
a.u?e:a.u}
function Dk(a){var b=a.ma,c;do{var d=a.Fd-a.u-a.o;if(a.o>=b+(b-262)){N.nb(a.window,a.window,b,b,0);a.ub-=b;a.o-=b;a.ta-=b;var e=c=a.hc;do{var f=a.head[--e];a.head[e]=f>=b?f-b:0}while(--c);e=c=b;do f=a.Ia[--e],a.Ia[e]=f>=b?f-b:0;while(--c);d+=b}if(a.K.na===0)break;e=a.K;c=a.window;f=a.o+a.u;var g=e.na;g>d&&(g=d);g===0?c=0:(e.na-=g,N.nb(c,e.input,e.hb,g,f),e.state.wrap===1?e.J=Jj(e.J,c,g,f):e.state.wrap===2&&(e.J=Kj(e.J,c,g,f)),e.hb+=g,e.kb+=g,c=g);a.u+=c;if(a.u+a.sa>=3)for(d=a.o-a.sa,a.M=a.window[d],
a.M=(a.M<<a.Ma^a.window[d+1])&a.La;a.sa&&!(a.M=(a.M<<a.Ma^a.window[d+3-1])&a.La,a.Ia[d&a.Xa]=a.head[a.M],a.head[a.M]=d,d++,a.sa--,a.u+a.sa<3););}while(a.u<262&&a.K.na!==0)}
function Ek(a,b){for(var c;;){if(a.u<262){Dk(a);if(a.u<262&&b===0)return 1;if(a.u===0)break}c=0;a.u>=3&&(a.M=(a.M<<a.Ma^a.window[a.o+3-1])&a.La,c=a.Ia[a.o&a.Xa]=a.head[a.M],a.head[a.M]=a.o);c!==0&&a.o-c<=a.ma-262&&(a.S=Ck(a,c));if(a.S>=3)if(c=wk(a,a.o-a.ub,a.S-3),a.u-=a.S,a.S<=a.Dc&&a.u>=3){a.S--;do a.o++,a.M=(a.M<<a.Ma^a.window[a.o+3-1])&a.La,a.Ia[a.o&a.Xa]=a.head[a.M],a.head[a.M]=a.o;while(--a.S!==0);a.o++}else a.o+=a.S,a.S=0,a.M=a.window[a.o],a.M=(a.M<<a.Ma^a.window[a.o+1])&a.La;else c=wk(a,0,
a.window[a.o]),a.u--,a.o++;if(c&&(Ak(a,!1),a.K.R===0))return 1}a.sa=a.o<2?a.o:2;return b===4?(Ak(a,!0),a.K.R===0?3:4):a.ya&&(Ak(a,!1),a.K.R===0)?1:2}
function Fk(a,b){for(var c,d;;){if(a.u<262){Dk(a);if(a.u<262&&b===0)return 1;if(a.u===0)break}c=0;a.u>=3&&(a.M=(a.M<<a.Ma^a.window[a.o+3-1])&a.La,c=a.Ia[a.o&a.Xa]=a.head[a.M],a.head[a.M]=a.o);a.wa=a.S;a.od=a.ub;a.S=2;c!==0&&a.wa<a.Dc&&a.o-c<=a.ma-262&&(a.S=Ck(a,c),a.S<=5&&(a.strategy===1||a.S===3&&a.o-a.ub>4096)&&(a.S=2));if(a.wa>=3&&a.S<=a.wa){d=a.o+a.u-3;c=wk(a,a.o-1-a.od,a.wa-3);a.u-=a.wa-1;a.wa-=2;do++a.o<=d&&(a.M=(a.M<<a.Ma^a.window[a.o+3-1])&a.La,a.Ia[a.o&a.Xa]=a.head[a.M],a.head[a.M]=a.o);
while(--a.wa!==0);a.fb=0;a.S=2;a.o++;if(c&&(Ak(a,!1),a.K.R===0))return 1}else if(a.fb){if((c=wk(a,0,a.window[a.o-1]))&&Ak(a,!1),a.o++,a.u--,a.K.R===0)return 1}else a.fb=1,a.o++,a.u--}a.fb&&(wk(a,0,a.window[a.o-1]),a.fb=0);a.sa=a.o<2?a.o:2;return b===4?(Ak(a,!0),a.K.R===0?3:4):a.ya&&(Ak(a,!1),a.K.R===0)?1:2}
function Gk(a,b){for(var c,d,e,f=a.window;;){if(a.u<=258){Dk(a);if(a.u<=258&&b===0)return 1;if(a.u===0)break}a.S=0;if(a.u>=3&&a.o>0&&(d=a.o-1,c=f[d],c===f[++d]&&c===f[++d]&&c===f[++d])){for(e=a.o+258;c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&d<e;);a.S=258-(e-d);a.S>a.u&&(a.S=a.u)}a.S>=3?(c=wk(a,1,a.S-3),a.u-=a.S,a.o+=a.S,a.S=0):(c=wk(a,0,a.window[a.o]),a.u--,a.o++);if(c&&(Ak(a,!1),a.K.R===0))return 1}a.sa=0;return b===4?(Ak(a,!0),a.K.R===0?3:4):
a.ya&&(Ak(a,!1),a.K.R===0)?1:2}
function Hk(a,b){for(var c;;){if(a.u===0&&(Dk(a),a.u===0)){if(b===0)return 1;break}a.S=0;c=wk(a,0,a.window[a.o]);a.u--;a.o++;if(c&&(Ak(a,!1),a.K.R===0))return 1}a.sa=0;return b===4?(Ak(a,!0),a.K.R===0?3:4):a.ya&&(Ak(a,!1),a.K.R===0)?1:2}
function Ik(a,b,c,d,e){this.ge=a;this.ue=b;this.ye=c;this.te=d;this.be=e}
var Jk;Jk=[new Ik(0,0,0,0,function(a,b){var c=65535;for(c>a.za-5&&(c=a.za-5);;){if(a.u<=1){Dk(a);if(a.u===0&&b===0)return 1;if(a.u===0)break}a.o+=a.u;a.u=0;var d=a.ta+c;if(a.o===0||a.o>=d)if(a.u=a.o-d,a.o=d,Ak(a,!1),a.K.R===0)return 1;if(a.o-a.ta>=a.ma-262&&(Ak(a,!1),a.K.R===0))return 1}a.sa=0;if(b===4)return Ak(a,!0),a.K.R===0?3:4;a.o>a.ta&&Ak(a,!1);return 1}),
new Ik(4,4,8,4,Ek),new Ik(4,5,16,8,Ek),new Ik(4,6,32,32,Ek),new Ik(4,4,16,16,Fk),new Ik(8,16,32,32,Fk),new Ik(8,16,128,128,Fk),new Ik(8,32,128,256,Fk),new Ik(32,128,258,1024,Fk),new Ik(32,258,258,4096,Fk)];
function Kk(){this.K=null;this.status=0;this.Z=null;this.wrap=this.pending=this.Ob=this.za=0;this.I=null;this.Ca=0;this.method=8;this.sb=-1;this.Xa=this.Rc=this.ma=0;this.window=null;this.Fd=0;this.head=this.Ia=null;this.ld=this.ed=this.strategy=this.level=this.Dc=this.kd=this.wa=this.u=this.ub=this.o=this.fb=this.od=this.S=this.ta=this.Ma=this.La=this.zc=this.hc=this.M=0;this.ra=new N.Ja(1146);this.bb=new N.Ja(122);this.ka=new N.Ja(78);yk(this.ra);yk(this.bb);yk(this.ka);this.Uc=this.cc=this.ic=
null;this.Ka=new N.Ja(16);this.aa=new N.Ja(573);yk(this.aa);this.qb=this.Na=0;this.depth=new N.Ja(573);yk(this.depth);this.ja=this.oa=this.sa=this.matches=this.zb=this.Pa=this.Gb=this.ya=this.Mb=this.Bc=0}
function Lk(a,b){if(!a||!a.state||b>5||b<0)return a?xk(a,-2):-2;var c=a.state;if(!a.output||!a.input&&a.na!==0||c.status===666&&b!==4)return xk(a,a.R===0?-5:-2);c.K=a;var d=c.sb;c.sb=b;if(c.status===42)if(c.wrap===2)a.J=0,R(c,31),R(c,139),R(c,8),c.I?(R(c,(c.I.text?1:0)+(c.I.Ta?2:0)+(c.I.extra?4:0)+(c.I.name?8:0)+(c.I.comment?16:0)),R(c,c.I.time&255),R(c,c.I.time>>8&255),R(c,c.I.time>>16&255),R(c,c.I.time>>24&255),R(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),R(c,c.I.os&255),c.I.extra&&c.I.extra.length&&
(R(c,c.I.extra.length&255),R(c,c.I.extra.length>>8&255)),c.I.Ta&&(a.J=Kj(a.J,c.Z,c.pending,0)),c.Ca=0,c.status=69):(R(c,0),R(c,0),R(c,0),R(c,0),R(c,0),R(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),R(c,3),c.status=113);else{var e=8+(c.Rc-8<<4)<<8;e|=(c.strategy>=2||c.level<2?0:c.level<6?1:c.level===6?2:3)<<6;c.o!==0&&(e|=32);c.status=113;Bk(c,e+(31-e%31));c.o!==0&&(Bk(c,a.J>>>16),Bk(c,a.J&65535));a.J=1}if(c.status===69)if(c.I.extra){for(e=c.pending;c.Ca<(c.I.extra.length&65535)&&(c.pending!==c.za||
(c.I.Ta&&c.pending>e&&(a.J=Kj(a.J,c.Z,c.pending-e,e)),zk(a),e=c.pending,c.pending!==c.za));)R(c,c.I.extra[c.Ca]&255),c.Ca++;c.I.Ta&&c.pending>e&&(a.J=Kj(a.J,c.Z,c.pending-e,e));c.Ca===c.I.extra.length&&(c.Ca=0,c.status=73)}else c.status=73;if(c.status===73)if(c.I.name){e=c.pending;do{if(c.pending===c.za&&(c.I.Ta&&c.pending>e&&(a.J=Kj(a.J,c.Z,c.pending-e,e)),zk(a),e=c.pending,c.pending===c.za)){var f=1;break}f=c.Ca<c.I.name.length?c.I.name.charCodeAt(c.Ca++)&255:0;R(c,f)}while(f!==0);c.I.Ta&&c.pending>
e&&(a.J=Kj(a.J,c.Z,c.pending-e,e));f===0&&(c.Ca=0,c.status=91)}else c.status=91;if(c.status===91)if(c.I.comment){e=c.pending;do{if(c.pending===c.za&&(c.I.Ta&&c.pending>e&&(a.J=Kj(a.J,c.Z,c.pending-e,e)),zk(a),e=c.pending,c.pending===c.za)){f=1;break}f=c.Ca<c.I.comment.length?c.I.comment.charCodeAt(c.Ca++)&255:0;R(c,f)}while(f!==0);c.I.Ta&&c.pending>e&&(a.J=Kj(a.J,c.Z,c.pending-e,e));f===0&&(c.status=103)}else c.status=103;c.status===103&&(c.I.Ta?(c.pending+2>c.za&&zk(a),c.pending+2<=c.za&&(R(c,a.J&
255),R(c,a.J>>8&255),a.J=0,c.status=113)):c.status=113);if(c.pending!==0){if(zk(a),a.R===0)return c.sb=-1,0}else if(a.na===0&&(b<<1)-(b>4?9:0)<=(d<<1)-(d>4?9:0)&&b!==4)return xk(a,-5);if(c.status===666&&a.na!==0)return xk(a,-5);if(a.na!==0||c.u!==0||b!==0&&c.status!==666){d=c.strategy===2?Hk(c,b):c.strategy===3?Gk(c,b):Jk[c.level].be(c,b);if(d===3||d===4)c.status=666;if(d===1||d===3)return a.R===0&&(c.sb=-1),0;if(d===2&&(b===1?(hk(c,2,3),ik(c,256,Vj),c.ja===16?(gk(c,c.oa),c.oa=0,c.ja=0):c.ja>=8&&
(c.Z[c.pending++]=c.oa&255,c.oa>>=8,c.ja-=8)):b!==5&&(hk(c,0,3),nk(c,0,0),b===3&&(yk(c.head),c.u===0&&(c.o=0,c.ta=0,c.sa=0))),zk(a),a.R===0))return c.sb=-1,0}if(b!==4)return 0;if(c.wrap<=0)return 1;c.wrap===2?(R(c,a.J&255),R(c,a.J>>8&255),R(c,a.J>>16&255),R(c,a.J>>24&255),R(c,a.kb&255),R(c,a.kb>>8&255),R(c,a.kb>>16&255),R(c,a.kb>>24&255)):(Bk(c,a.J>>>16),Bk(c,a.J&65535));zk(a);c.wrap>0&&(c.wrap=-c.wrap);return c.pending!==0?0:1}
;var Mk={};Mk=function(){this.input=null;this.kb=this.na=this.hb=0;this.output=null;this.Oc=this.R=this.wb=0;this.msg="";this.state=null;this.wc=2;this.J=0};var Nk=Object.prototype.toString;
function Ok(a){if(!(this instanceof Ok))return new Ok(a);a=this.options=N.assign({level:-1,method:8,chunkSize:16384,windowBits:15,memLevel:8,strategy:0,to:""},a||{});a.raw&&a.windowBits>0?a.windowBits=-a.windowBits:a.gzip&&a.windowBits>0&&a.windowBits<16&&(a.windowBits+=16);this.err=0;this.msg="";this.ended=!1;this.chunks=[];this.K=new Mk;this.K.R=0;var b=this.K;var c=a.level,d=a.method,e=a.windowBits,f=a.memLevel,g=a.strategy;if(b){var h=1;c===-1&&(c=6);e<0?(h=0,e=-e):e>15&&(h=2,e-=16);if(f<1||f>
9||d!==8||e<8||e>15||c<0||c>9||g<0||g>4)b=xk(b,-2);else{e===8&&(e=9);var k=new Kk;b.state=k;k.K=b;k.wrap=h;k.I=null;k.Rc=e;k.ma=1<<k.Rc;k.Xa=k.ma-1;k.zc=f+7;k.hc=1<<k.zc;k.La=k.hc-1;k.Ma=~~((k.zc+3-1)/3);k.window=new N.lb(k.ma*2);k.head=new N.Ja(k.hc);k.Ia=new N.Ja(k.ma);k.Mb=1<<f+6;k.za=k.Mb*4;k.Z=new N.lb(k.za);k.Gb=1*k.Mb;k.Bc=3*k.Mb;k.level=c;k.strategy=g;k.method=d;if(b&&b.state){b.kb=b.Oc=0;b.wc=2;c=b.state;c.pending=0;c.Ob=0;c.wrap<0&&(c.wrap=-c.wrap);c.status=c.wrap?42:113;b.J=c.wrap===2?
0:1;c.sb=0;if(!vk){d=Array(16);for(f=g=0;f<28;f++)for(Zj[f]=g,e=0;e<1<<Rj[f];e++)Yj[g++]=f;Yj[g-1]=f;for(f=g=0;f<16;f++)for(ak[f]=g,e=0;e<1<<Sj[f];e++)Xj[g++]=f;for(g>>=7;f<30;f++)for(ak[f]=g<<7,e=0;e<1<<Sj[f]-7;e++)Xj[256+g++]=f;for(e=0;e<=15;e++)d[e]=0;for(e=0;e<=143;)Vj[e*2+1]=8,e++,d[8]++;for(;e<=255;)Vj[e*2+1]=9,e++,d[9]++;for(;e<=279;)Vj[e*2+1]=7,e++,d[7]++;for(;e<=287;)Vj[e*2+1]=8,e++,d[8]++;kk(Vj,287,d);for(e=0;e<30;e++)Wj[e*2+1]=5,Wj[e*2]=jk(e,5);ck=new bk(Vj,Rj,257,286,15);dk=new bk(Wj,
Sj,0,30,15);ek=new bk([],Tj,0,19,7);vk=!0}c.ic=new fk(c.ra,ck);c.cc=new fk(c.bb,dk);c.Uc=new fk(c.ka,ek);c.oa=0;c.ja=0;lk(c);c=0}else c=xk(b,-2);c===0&&(b=b.state,b.Fd=2*b.ma,yk(b.head),b.Dc=Jk[b.level].ue,b.ed=Jk[b.level].ge,b.ld=Jk[b.level].ye,b.kd=Jk[b.level].te,b.o=0,b.ta=0,b.u=0,b.sa=0,b.S=b.wa=2,b.fb=0,b.M=0);b=c}}else b=-2;if(b!==0)throw Error(Pj[b]);a.header&&(b=this.K)&&b.state&&b.state.wrap===2&&(b.state.I=a.header);if(a.dictionary){var l;typeof a.dictionary==="string"?l=Ij(a.dictionary):
Nk.call(a.dictionary)==="[object ArrayBuffer]"?l=new Uint8Array(a.dictionary):l=a.dictionary;a=this.K;f=l;g=f.length;if(a&&a.state)if(l=a.state,b=l.wrap,b===2||b===1&&l.status!==42||l.u)b=-2;else{b===1&&(a.J=Jj(a.J,f,g,0));l.wrap=0;g>=l.ma&&(b===0&&(yk(l.head),l.o=0,l.ta=0,l.sa=0),c=new N.lb(l.ma),N.nb(c,f,g-l.ma,l.ma,0),f=c,g=l.ma);c=a.na;d=a.hb;e=a.input;a.na=g;a.hb=0;a.input=f;for(Dk(l);l.u>=3;){f=l.o;g=l.u-2;do l.M=(l.M<<l.Ma^l.window[f+3-1])&l.La,l.Ia[f&l.Xa]=l.head[l.M],l.head[l.M]=f,f++;while(--g);
l.o=f;l.u=2;Dk(l)}l.o+=l.u;l.ta=l.o;l.sa=l.u;l.u=0;l.S=l.wa=2;l.fb=0;a.hb=d;a.input=e;a.na=c;l.wrap=b;b=0}else b=-2;if(b!==0)throw Error(Pj[b]);this.Bg=!0}}
Ok.prototype.push=function(a,b){var c=this.K,d=this.options.chunkSize;if(this.ended)return!1;var e=b===~~b?b:b===!0?4:0;typeof a==="string"?c.input=Ij(a):Nk.call(a)==="[object ArrayBuffer]"?c.input=new Uint8Array(a):c.input=a;c.hb=0;c.na=c.input.length;do{c.R===0&&(c.output=new N.lb(d),c.wb=0,c.R=d);a=Lk(c,e);if(a!==1&&a!==0)return Pk(this,a),this.ended=!0,!1;if(c.R===0||c.na===0&&(e===4||e===2))if(this.options.to==="string"){var f=N.Nc(c.output,c.wb);b=f;f=f.length;if(f<65537&&(b.subarray&&Hj||!b.subarray))b=
String.fromCharCode.apply(null,N.Nc(b,f));else{for(var g="",h=0;h<f;h++)g+=String.fromCharCode(b[h]);b=g}this.chunks.push(b)}else b=N.Nc(c.output,c.wb),this.chunks.push(b)}while((c.na>0||c.R===0)&&a!==1);if(e===4)return(c=this.K)&&c.state?(d=c.state.status,d!==42&&d!==69&&d!==73&&d!==91&&d!==103&&d!==113&&d!==666?a=xk(c,-2):(c.state=null,a=d===113?xk(c,-3):0)):a=-2,Pk(this,a),this.ended=!0,a===0;e===2&&(Pk(this,0),c.R=0);return!0};
function Pk(a,b){b===0&&(a.result=a.options.to==="string"?a.chunks.join(""):N.cd(a.chunks));a.chunks=[];a.err=b;a.msg=a.K.msg}
function Qk(a,b){b=b||{};b.gzip=!0;b=new Ok(b);b.push(a,!0);if(b.err)throw b.msg||Pj[b.err];return b.result}
;function Rk(a){if(!a)return null;a=a.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue;var b;a?b=mb(a):b=null;return b}
;function Sk(a){return mb(a===null?"null":a===void 0?"undefined":a)}
;function Tk(a){this.name=a}
;var Uk=new Tk("rawColdConfigGroup");var Vk=new Tk("rawHotConfigGroup");function Wk(a){this.D=I(a)}
y(Wk,L);var Xk=new Tk("continuationCommand");var Yk=new Tk("webCommandMetadata");var Zk=new Tk("signalServiceEndpoint");var $k={Ef:"EMBEDDED_PLAYER_MODE_UNKNOWN",Bf:"EMBEDDED_PLAYER_MODE_DEFAULT",Df:"EMBEDDED_PLAYER_MODE_PFP",Cf:"EMBEDDED_PLAYER_MODE_PFL"};var al=new Tk("feedbackEndpoint");function bl(a){this.D=I(a)}
y(bl,L);bl.prototype.setTrackingParams=function(a){if(a!=null)if(typeof a==="string")a=a?new We(a,Te):Ue||(Ue=new We(null,Te));else if(a.constructor!==We)if(Se(a))a=a.length?new We(new Uint8Array(a),Te):Ue||(Ue=new We(null,Te));else throw Error();return K(this,1,a)};var Yi={gg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNKNOWN",Of:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_FOR_TESTING",Wf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_RESUME_TO_HOME_TTL",ag:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_START_TO_SHORTS_ANALYSIS_SLICE",Lf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DEVICE_LAYER_SLICE",fg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNIFIED_LAYER_SLICE",hg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_VISITOR_LAYER_SLICE",Zf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SHOW_SHEET_COMMAND_HANDLER_BLOCK",
jg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_MIGRATED_COMPONENT",ig:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_CHANNEL_NAME_TOOLTIP",Xf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATION_LOCK_SUPPORTED",dg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_THEATER_MODE_ENABLED",ng:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_PIN_SUGGESTION",mg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_LONG_PRESS_EDU_TOAST",lg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_AMBIENT",eg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TIME_WATCHED_PANEL",
Yf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SEARCH_FROM_SEARCH_BAR_OVERLAY",og:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_VOICE_SEARCH_EDU_TOAST",cg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SUGGESTED_LANGUAGE_SELECTED",pg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_TRIGGER_SHORTS_PIP",Pf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IN_ZP_VOICE_CRASHY_SET",Sf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_SUPPRESSED",Rf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_ALLOWED",Uf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_PULL_TO_REFRESH_ATTEMPT",
kg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_BLOCK_KABUKI",Vf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_TALL_SCREEN",Tf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_NORMAL_SCREEN",If:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_ENABLED",Hf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_DISABLED",Jf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_AUTOPLAY_ENABLED",Kf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_CAST_MATCH_OCCURRED",Mf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_ELIGIBLE",Nf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ENDSCREEN_TRIGGERED",
Qf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_POSTPLAY_TRIGGERED"};var cl=new Tk("webPlayerShareEntityServiceEndpoint");var dl=new Tk("playlistEditEndpoint");var el=new Tk("modifyChannelNotificationPreferenceEndpoint");var fl=new Tk("unsubscribeEndpoint");var gl=new Tk("subscribeEndpoint");function hl(){var a=il;E("yt.ads.biscotti.getId_")||D("yt.ads.biscotti.getId_",a)}
function jl(a){D("yt.ads.biscotti.lastId_",a)}
;function kl(a,b){b.length>1?a[b[0]]=b[1]:b.length===1&&Object.assign(a,b[0])}
;var ll=C.window,ml,nl,ol=(ll==null?void 0:(ml=ll.yt)==null?void 0:ml.config_)||(ll==null?void 0:(nl=ll.ytcfg)==null?void 0:nl.data_)||{};D("yt.config_",ol);function pl(){kl(ol,arguments)}
function T(a,b){return a in ol?ol[a]:b}
function ql(a){var b=ol.EXPERIMENT_FLAGS;return b?b[a]:void 0}
;var rl=[];function sl(a){rl.forEach(function(b){return b(a)})}
function tl(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){ul(b)}}:a}
function ul(a){var b=E("yt.logging.errors.log");b?b(a,"ERROR",void 0,void 0,void 0,void 0,void 0):(b=T("ERRORS",[]),b.push([a,"ERROR",void 0,void 0,void 0,void 0,void 0]),pl("ERRORS",b));sl(a)}
function vl(a,b,c,d,e){var f=E("yt.logging.errors.log");f?f(a,"WARNING",b,c,d,void 0,e):(f=T("ERRORS",[]),f.push([a,"WARNING",b,c,d,void 0,e]),pl("ERRORS",f))}
;var wl=/^[\w.]*$/,xl={q:!0,search_query:!0};function yl(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(f.length===1&&f[0]||f.length===2)try{var g=zl(f[0]||""),h=zl(f[1]||"");if(g in c){var k=c[g];Array.isArray(k)?Jb(k,h):c[g]=[k,h]}else c[g]=h}catch(t){var l=t,m=f[0],p=String(yl);l.args=[{key:m,value:f[1],query:a,method:Al===p?"unchanged":p}];xl.hasOwnProperty(m)||vl(l)}}return c}
var Al=String(yl);function Bl(a){var b=[];Kb(a,function(c,d){var e=encodeURIComponent(String(d));c=Array.isArray(c)?c:[c];Db(c,function(f){f==""?b.push(e):b.push(e+"="+encodeURIComponent(String(f)))})});
return b.join("&")}
function Cl(a){a.charAt(0)==="?"&&(a=a.substring(1));return yl(a,"&")}
function Dl(a){return a.indexOf("?")!==-1?(a=(a||"").split("#")[0],a=a.split("?",2),Cl(a.length>1?a[1]:a[0])):{}}
function El(a,b,c){var d=a.split("#",2);a=d[0];d=d.length>1?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=Cl(e[1]||"");for(var f in b)!c&&e!==null&&f in e||(e[f]=b[f]);return tc(a,e)+d}
function Fl(a){if(!b)var b=window.location.href;var c=nc(1,a),d=oc(a);c&&d?(a=a.match(lc),b=b.match(lc),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?oc(b)===d&&(Number(nc(4,b))||null)===(Number(nc(4,a))||null):!0;return a}
function zl(a){return a&&a.match(wl)?a:decodeURIComponent(a.replace(/\+/g," "))}
;function Gl(a){var b=Hl;a=a===void 0?E("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=Wi;e.flash="0";a:{try{var f=b.h.top.location.href}catch(Ka){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();var g=g===void 0?Oi:g;try{var h=g.history.length}catch(Ka){h=0}e.u_his=h;var k;e.u_h=(k=Oi.screen)==null?void 0:k.height;var l;e.u_w=(l=Oi.screen)==null?void 0:l.width;var m;e.u_ah=(m=Oi.screen)==null?void 0:m.availHeight;var p;e.u_aw=
(p=Oi.screen)==null?void 0:p.availWidth;var t;e.u_cd=(t=Oi.screen)==null?void 0:t.colorDepth}catch(Ka){}h=b.h;try{var r=h.screenX;var v=h.screenY}catch(Ka){}try{var u=h.outerWidth;var z=h.outerHeight}catch(Ka){}try{var F=h.innerWidth;var J=h.innerHeight}catch(Ka){}try{var O=h.screenLeft;var S=h.screenTop}catch(Ka){}try{F=h.innerWidth,J=h.innerHeight}catch(Ka){}try{var da=h.screen.availWidth;var va=h.screen.availTop}catch(Ka){}r=[O,S,r,v,da,va,u,z,F,J];try{var P=(b.h.top||window).document,ea=P.compatMode==
"CSS1Compat"?P.documentElement:P.body;var na=(new Ed(ea.clientWidth,ea.clientHeight)).round()}catch(Ka){na=new Ed(-12245933,-12245933)}P=na;na={};var La=La===void 0?C:La;ea=new dj;"SVGElement"in La&&"createElementNS"in La.document&&ea.set(0);v=Ti();v["allow-top-navigation-by-user-activation"]&&ea.set(1);v["allow-popups-to-escape-sandbox"]&&ea.set(2);La.crypto&&La.crypto.subtle&&ea.set(3);"TextDecoder"in La&&"TextEncoder"in La&&ea.set(4);La=ej(ea);na.bc=La;na.bih=P.height;na.biw=P.width;na.brdim=r.join();
b=b.i;b=(na.vis=b.prerendering?3:{visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""]||0,na.wgl=!!Oi.WebGLRenderingContext,na);c=d.call(c,e,b);c.ca_type="image";a&&(c.bid=a);return c}
var Hl=new function(){var a=window.document;this.h=window;this.i=a};
D("yt.ads_.signals_.getAdSignalsString",function(a){return Bl(Gl(a))});Za();navigator.userAgent.indexOf(" (CrKey ");var Il="XMLHttpRequest"in C?function(){return new XMLHttpRequest}:null;
function Jl(){if(!Il)return null;var a=Il();return"open"in a?a:null}
function Kl(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function Ll(a,b){typeof a==="function"&&(a=tl(a));return window.setTimeout(a,b)}
;var Ml="client_dev_domain client_dev_expflag client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");[].concat(la(Ml),["client_dev_set_cookie"]);function U(a){a=Nl(a);return typeof a==="string"&&a==="false"?!1:!!a}
function Ol(a,b){a=Nl(a);return a===void 0&&b!==void 0?b:Number(a||0)}
function Nl(a){return T("EXPERIMENT_FLAGS",{})[a]}
function Pl(){for(var a=[],b=T("EXPERIMENTS_FORCED_FLAGS",{}),c=x(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=T("EXPERIMENT_FLAGS",{});d=x(Object.keys(c));for(var e=d.next();!e.done;e=d.next())e=e.value,e.startsWith("force_")&&b[e]===void 0&&a.push({key:e,value:String(c[e])});return a}
;var Ql={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM","X-Goog-AuthUser":"SESSION_INDEX","X-Goog-PageId":"DELEGATED_SESSION_ID"},Rl="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(la(Ml)),Sl=!1;
function Tl(a,b,c,d,e,f,g,h){function k(){(l&&"readyState"in l?l.readyState:0)===4&&b&&tl(b)(l)}
c=c===void 0?"GET":c;d=d===void 0?"":d;h=h===void 0?!1:h;var l=Jl();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",k,!1):l.onreadystatechange=k;U("debug_forward_web_query_parameters")&&(a=Ul(a));l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c=c==="POST"&&(window.FormData===void 0||!(d instanceof FormData));if(e=Vl(a,e))for(var m in e)l.setRequestHeader(m,e[m]),"content-type"===m.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
if(h&&"setAttributionReporting"in XMLHttpRequest.prototype){a={eventSourceEligible:!0,triggerEligible:!1};try{l.setAttributionReporting(a)}catch(p){vl(p)}}l.send(d);return l}
function Vl(a,b){b=b===void 0?{}:b;var c=Fl(a),d=T("INNERTUBE_CLIENT_NAME"),e=U("web_ajax_ignore_global_headers_if_set"),f;for(f in Ql){var g=T(Ql[f]),h=f==="X-Goog-AuthUser"||f==="X-Goog-PageId";f!=="X-Goog-Visitor-Id"||g||(g=T("VISITOR_DATA"));var k;if(!(k=!g)){if(!(k=c||(oc(a)?!1:!0))){k=a;var l;if(l=U("add_auth_headers_to_remarketing_google_dot_com_ping")&&f==="Authorization"&&(d==="TVHTML5"||d==="TVHTML5_UNPLUGGED"||d==="TVHTML5_SIMPLY"))l=oc(k),l=l!==null?l.split(".").reverse():null,l=l===null?
!1:l[1]==="google"?!0:l[2]==="google"?l[0]==="au"&&l[1]==="com"?!0:l[0]==="uk"&&l[1]==="co"?!0:!1:!1;l&&(k=mc(nc(5,k))||"",k=k.split("/"),k="/"+(k.length>1?k[1]:""),l=k==="/pagead");k=l?!0:!1}k=!k}k||e&&b[f]!==void 0||d==="TVHTML5_UNPLUGGED"&&h||(b[f]=g)}"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!oc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());if(c||!oc(a)){try{var m=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(p){}m&&
(b["X-YouTube-Time-Zone"]=m)}document.location.hostname.endsWith("youtubeeducation.com")||!c&&oc(a)||(b["X-YouTube-Ad-Signals"]=Bl(Gl()));return b}
function Wl(a,b){b.method="POST";b.postParams||(b.postParams={});return Xl(a,b)}
function Xl(a,b){var c=b.format||"JSON";a=Yl(a,b);var d=Zl(a,b),e=!1,f=$l(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);var l=Kl(k),m=null,p=400<=k.status&&k.status<500,t=500<=k.status&&k.status<600;if(l||p||t)m=am(a,c,k,b.convertToSafeHtml);l&&(l=bm(c,k,m));m=m||{};p=b.context||C;l?b.onSuccess&&b.onSuccess.call(p,k,m):b.onError&&b.onError.call(p,k,m);b.onFinish&&b.onFinish.call(p,k,m)}},b.method,d,b.headers,b.responseType,b.withCredentials);
d=b.timeout||0;if(b.onTimeout&&d>0){var g=b.onTimeout;var h=Ll(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||C,f))},d)}return f}
function Yl(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=T("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=El(a,b||{},!0);return a}
function Zl(a,b){var c=T("XSRF_FIELD_NAME"),d=T("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=T("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||oc(a)&&!b.withCredentials&&oc(a)!==document.location.hostname||b.method!=="POST"||h&&h!=="application/x-www-form-urlencoded"||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(U("ajax_parse_query_data_only_when_filled")&&f&&Object.keys(f).length>0||f)&&typeof e==="string"&&(e=Cl(e),Ub(e,f),e=b.postBodyFormat&&b.postBodyFormat===
"JSON"?JSON.stringify(e):sc(e));f=e||f&&!Nb(f);!Sl&&f&&b.method!=="POST"&&(Sl=!0,ul(Error("AJAX request with postData should use POST")));return e}
function am(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,vl(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&a.indexOf("json")>=0&&(f.substring(0,5)===")]}'\n"&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?cm(a):null)e={},Db(a.getElementsByTagName("*"),function(g){e[g.tagName]=dm(g)})}d&&em(e);
return e}
function em(a){if(Ra(a))for(var b in a){var c;(c=b==="html_content")||(c=b.length-5,c=c>=0&&b.indexOf("_html",c)==c);if(c){c=b;var d=a[b],e=fb();d=e?e.createHTML(d):d;a[c]=new Vb(d)}else em(a[b])}}
function bm(a,b,c){if(b&&b.status===204)return!0;switch(a){case "JSON":return!!c;case "XML":return Number(c&&c.return_code)===0;case "RAW":return!0;default:return!!c}}
function cm(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&a.length>0?a[0]:null:null}
function dm(a){var b="";Db(a.childNodes,function(c){b+=c.nodeValue});
return b}
function Ul(a){var b=window.location.search,c=oc(a);U("debug_handle_relative_url_for_query_forward_killswitch")||!c&&Fl(a)&&(c=document.location.hostname);var d=mc(nc(5,a));d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=Cl(b),f={};Db(Rl,function(g){e[g]&&(f[g]=e[g])});
return El(a,f||{},!1)}
var $l=Tl;var fm=[{Ec:function(a){return"Cannot read property '"+a.key+"'"},
kc:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{Ec:function(a){return"Cannot call '"+a.key+"'"},
kc:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{Ec:function(a){return a.key+" is not defined"},
kc:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var hm={Va:[],Sa:[{callback:gm,weight:500}]};function gm(a){if(a.name==="JavaException")return!0;a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function im(){this.Sa=[];this.Va=[]}
var jm;function km(){if(!jm){var a=jm=new im;a.Va.length=0;a.Sa.length=0;hm.Va&&a.Va.push.apply(a.Va,hm.Va);hm.Sa&&a.Sa.push.apply(a.Sa,hm.Sa)}return jm}
;var lm=new M;function mm(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=nm(b);if(e===Infinity)break;var f=e>>3;switch(e&7){case 0:e=nm(b);if(f===2)return e;break;case 1:if(f===2)return;d+=8;break;case 2:e=nm(b);if(f===2)return a.substr(d,e);d+=e;break;case 5:if(f===2)return;d+=4;break;default:return}}while(d<c)}
function nm(a){var b=a(),c=b&127;if(b<128)return c;b=a();c|=(b&127)<<7;if(b<128)return c;b=a();c|=(b&127)<<14;if(b<128)return c;b=a();return b<128?c|(b&127)<<21:Infinity}
;function om(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=pm(d,a[d],b,c),e>500));d++);d=e}else if(typeof a==="object")for(e in a){if(a[e]){var f=e;var g=a[e],h=b,k=c;f=typeof g!=="string"||f!=="clickTrackingParams"&&f!=="trackingParams"?0:(g=mm(atob(g.replace(/-/g,"+").replace(/_/g,"/"))))?pm(f+".ve",g,h,k):0;d+=f;d+=pm(e,a[e],b,c);if(d>500)break}}else c[b]=qm(a),d+=c[b].length;else c[b]=qm(a),d+=c[b].length;return d}
function pm(a,b,c,d){c+="."+a;a=qm(b);d[c]=a;return c.length+a.length}
function qm(a){try{return(typeof a==="string"?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;function rm(a){var b=this;this.i=void 0;this.h=!1;a.addEventListener("beforeinstallprompt",function(c){c.preventDefault();b.i=c});
a.addEventListener("appinstalled",function(){b.h=!0},{once:!0})}
function sm(){if(!C.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return C.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":C.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":C.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":C.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function tm(){this.df=!0}
function um(){tm.h||(tm.h=new tm);return tm.h}
function wm(a,b){a={};var c=[];"SESSION_ID"in ol&&c.push({key:"u",value:T("SESSION_ID")});if(c=Ch(c))a.Authorization=c,c=b=b==null?void 0:b.sessionIndex,c===void 0&&(c=Number(T("SESSION_INDEX",0)),c=isNaN(c)?0:c),U("voice_search_auth_header_removal")||(a["X-Goog-AuthUser"]=c.toString()),"INNERTUBE_HOST_OVERRIDE"in ol||(a["X-Origin"]=window.location.origin),b===void 0&&"DELEGATED_SESSION_ID"in ol&&(a["X-Goog-PageId"]=T("DELEGATED_SESSION_ID"));return a}
;var xm={identityType:"UNAUTHENTICATED_IDENTITY_TYPE_UNKNOWN"};function ym(a,b,c,d,e){yh.set(""+a,b,{Nb:c,path:"/",domain:d===void 0?"youtube.com":d,secure:e===void 0?!1:e})}
function zm(a){return yh.get(""+a,void 0)}
function Am(a,b,c){yh.remove(""+a,b===void 0?"/":b,c===void 0?"youtube.com":c)}
function Bm(){if(U("embeds_web_enable_cookie_detection_fix")){if(!C.navigator.cookieEnabled)return!1}else if(!yh.isEnabled())return!1;if(yh.h.cookie)return!0;U("embeds_web_enable_cookie_detection_fix")?yh.set("TESTCOOKIESENABLED","1",{Nb:60,Le:"none",secure:!0}):yh.set("TESTCOOKIESENABLED","1",{Nb:60});if(yh.get("TESTCOOKIESENABLED")!=="1")return!1;yh.remove("TESTCOOKIESENABLED");return!0}
;var Cm=E("ytglobal.prefsUserPrefsPrefs_")||{};D("ytglobal.prefsUserPrefsPrefs_",Cm);function Dm(){this.h=T("ALT_PREF_COOKIE_NAME","PREF");this.i=T("ALT_PREF_COOKIE_DOMAIN","youtube.com");var a=zm(this.h);a&&this.parse(a)}
var Em;function Fm(){Em||(Em=new Dm);return Em}
n=Dm.prototype;n.get=function(a,b){Gm(a);Hm(a);a=Cm[a]!==void 0?Cm[a].toString():null;return a!=null?a:b?b:""};
n.set=function(a,b){Gm(a);Hm(a);if(b==null)throw Error("ExpectedNotNull");Cm[a]=b.toString()};
function Im(a){return!!((Jm("f"+(Math.floor(a/31)+1))||0)&1<<a%31)}
n.remove=function(a){Gm(a);Hm(a);delete Cm[a]};
n.clear=function(){for(var a in Cm)delete Cm[a]};
function Hm(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function Gm(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function Jm(a){a=Cm[a]!==void 0?Cm[a].toString():null;return a!=null&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
n.parse=function(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Cm[d]=c.toString())}};var Km={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},Lm={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};
function Mm(){var a=C.navigator;return a?a.connection:void 0}
function Nm(){var a=Mm();if(a){var b=Km[a.type||"unknown"]||"CONN_UNKNOWN";a=Km[a.effectiveType||"unknown"]||"CONN_UNKNOWN";b==="CONN_CELLULAR_UNKNOWN"&&a!=="CONN_UNKNOWN"&&(b=a);if(b!=="CONN_UNKNOWN")return b;if(a!=="CONN_UNKNOWN")return a}}
function Om(){var a=Mm();if(a!=null&&a.effectiveType)return Lm.hasOwnProperty(a.effectiveType)?Lm[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN"}
;function V(a){var b=B.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(la(b))}
y(V,Error);function Pm(){try{return Qm(),!0}catch(a){return!1}}
function Qm(a){if(T("DATASYNC_ID")!==void 0)return T("DATASYNC_ID");throw new V("Datasync ID not set",a===void 0?"unknown":a);}
;function Rm(){}
function Sm(a,b){return cj.ab(a,0,b)}
Rm.prototype.pa=function(a,b){return this.ab(a,1,b)};
Rm.prototype.Db=function(a){var b=E("yt.scheduler.instance.addImmediateJob");b?b(a):a()};var Tm=Ol("web_emulated_idle_callback_delay",300),Um=1E3/60-3,Vm=[8,5,4,3,2,1,0];
function Wm(a){a=a===void 0?{}:a;G.call(this);this.i=[];this.j={};this.da=this.h=0;this.ba=this.m=!1;this.P=[];this.W=this.ga=!1;for(var b=x(Vm),c=b.next();!c.done;c=b.next())this.i[c.value]=[];this.l=0;this.vc=a.timeout||1;this.H=Um;this.A=0;this.xa=this.Ae.bind(this);this.uc=this.gf.bind(this);this.Za=this.Ld.bind(this);this.Cb=this.he.bind(this);this.Tb=this.De.bind(this);this.Ga=!!window.requestIdleCallback&&!!window.cancelIdleCallback&&!U("disable_scheduler_requestIdleCallback");(this.ia=a.useRaf!==
!1&&!!window.requestAnimationFrame)&&document.addEventListener("visibilitychange",this.xa)}
y(Wm,G);n=Wm.prototype;n.Db=function(a){var b=Za();Xm(this,a);a=Za()-b;this.m||(this.H-=a)};
n.ab=function(a,b,c){++this.da;if(b===10)return this.Db(a),this.da;var d=this.da;this.j[d]=a;this.m&&!c?this.P.push({id:d,priority:b}):(this.i[b].push(d),this.ba||this.m||(this.h!==0&&Ym(this)!==this.A&&this.stop(),this.start()));return d};
n.qa=function(a){delete this.j[a]};
function Zm(a){a.P.length=0;for(var b=5;b>=0;b--)a.i[b].length=0;a.i[8].length=0;a.j={};a.stop()}
n.isHidden=function(){return!!document.hidden||!1};
function $m(a){return!a.isHidden()&&a.ia}
function Ym(a){if(a.i[8].length){if(a.W)return 4;if($m(a))return 3}for(var b=5;b>=a.l;b--)if(a.i[b].length>0)return b>0?$m(a)?3:2:1;return 0}
n.Ha=function(a){var b=E("yt.logging.errors.log");b&&b(a)};
function Xm(a,b){try{b()}catch(c){a.Ha(c)}}
function an(a){for(var b=x(Vm),c=b.next();!c.done;c=b.next())if(a.i[c.value].length)return!0;return!1}
n.he=function(a){var b=void 0;a&&(b=a.timeRemaining());this.ga=!0;bn(this,b);this.ga=!1};
n.gf=function(){bn(this)};
n.Ld=function(){cn(this)};
n.De=function(a){this.W=!0;var b=Ym(this);b===4&&b!==this.A&&(this.stop(),this.start());bn(this,void 0,a);this.W=!1};
n.Ae=function(){this.isHidden()||cn(this);this.h&&(this.stop(),this.start())};
function cn(a){a.stop();a.m=!0;for(var b=Za(),c=a.i[8];c.length;){var d=c.shift(),e=a.j[d];delete a.j[d];e&&Xm(a,e)}dn(a);a.m=!1;an(a)&&a.start();b=Za()-b;a.H-=b}
function dn(a){for(var b=0,c=a.P.length;b<c;b++){var d=a.P[b];a.i[d.priority].push(d.id)}a.P.length=0}
function bn(a,b,c){a.W&&a.A===4&&a.h||a.stop();a.m=!0;b=Za()+(b||a.H);for(var d=a.i[5];d.length;){var e=d.shift(),f=a.j[e];delete a.j[e];if(f){e=a;try{f(c)}catch(l){e.Ha(l)}}}for(d=a.i[4];d.length;)c=d.shift(),f=a.j[c],delete a.j[c],f&&Xm(a,f);d=a.ga?0:1;d=a.l>d?a.l:d;if(!(Za()>=b)){do{a:{c=a;f=d;for(e=3;e>=f;e--)for(var g=c.i[e];g.length;){var h=g.shift(),k=c.j[h];delete c.j[h];if(k){c=k;break a}}c=null}c&&Xm(a,c)}while(c&&Za()<b)}a.m=!1;dn(a);a.H=Um;an(a)&&a.start()}
n.start=function(){this.ba=!1;if(this.h===0)switch(this.A=Ym(this),this.A){case 1:var a=this.Cb;this.h=this.Ga?window.requestIdleCallback(a,{timeout:3E3}):window.setTimeout(a,Tm);break;case 2:this.h=window.setTimeout(this.uc,this.vc);break;case 3:this.h=window.requestAnimationFrame(this.Tb);break;case 4:this.h=window.setTimeout(this.Za,0)}};
n.pause=function(){this.stop();this.ba=!0};
n.stop=function(){if(this.h){switch(this.A){case 1:var a=this.h;this.Ga?window.cancelIdleCallback(a):window.clearTimeout(a);break;case 2:case 4:window.clearTimeout(this.h);break;case 3:window.cancelAnimationFrame(this.h)}this.h=0}};
n.U=function(){Zm(this);this.stop();this.ia&&document.removeEventListener("visibilitychange",this.xa);G.prototype.U.call(this)};var en=E("yt.scheduler.instance.timerIdMap_")||{},fn=Ol("kevlar_tuner_scheduler_soft_state_timer_ms",800),gn=0,hn=0;function jn(){var a=E("ytglobal.schedulerInstanceInstance_");if(!a||a.V)a=new Wm(T("scheduler")||{}),D("ytglobal.schedulerInstanceInstance_",a);return a}
function kn(){ln();var a=E("ytglobal.schedulerInstanceInstance_");a&&(Cc(a),D("ytglobal.schedulerInstanceInstance_",null))}
function ln(){Zm(jn());for(var a in en)en.hasOwnProperty(a)&&delete en[Number(a)]}
function mn(a,b,c){if(!c)return c=c===void 0,-jn().ab(a,b,c);var d=window.setTimeout(function(){var e=jn().ab(a,b);en[d]=e},c);
return d}
function nn(a){jn().Db(a)}
function on(a){var b=jn();if(a<0)b.qa(-a);else{var c=en[a];c?(b.qa(c),delete en[a]):window.clearTimeout(a)}}
function pn(){qn()}
function qn(){window.clearTimeout(gn);jn().start()}
function rn(){jn().pause();window.clearTimeout(gn);gn=window.setTimeout(pn,fn)}
function sn(){window.clearTimeout(hn);hn=window.setTimeout(function(){tn(0)},fn)}
function tn(a){sn();var b=jn();b.l=a;b.start()}
function un(a){sn();var b=jn();b.l>a&&(b.l=a,b.start())}
function vn(){window.clearTimeout(hn);var a=jn();a.l=0;a.start()}
;function wn(){Rm.apply(this,arguments)}
y(wn,Rm);function xn(){wn.h||(wn.h=new wn);return wn.h}
wn.prototype.ab=function(a,b,c){c!==void 0&&Number.isNaN(Number(c))&&(c=void 0);var d=E("yt.scheduler.instance.addJob");return d?d(a,b,c):c===void 0?(a(),NaN):Ll(a,c||0)};
wn.prototype.qa=function(a){if(a===void 0||!Number.isNaN(Number(a))){var b=E("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
wn.prototype.start=function(){var a=E("yt.scheduler.instance.start");a&&a()};
wn.prototype.pause=function(){var a=E("yt.scheduler.instance.pause");a&&a()};
var cj=xn();
U("web_scheduler_auto_init")&&!E("yt.scheduler.initialized")&&(D("yt.scheduler.instance.dispose",kn),D("yt.scheduler.instance.addJob",mn),D("yt.scheduler.instance.addImmediateJob",nn),D("yt.scheduler.instance.cancelJob",on),D("yt.scheduler.instance.cancelAllJobs",ln),D("yt.scheduler.instance.start",qn),D("yt.scheduler.instance.pause",rn),D("yt.scheduler.instance.setPriorityThreshold",tn),D("yt.scheduler.instance.enablePriorityThreshold",un),D("yt.scheduler.instance.clearPriorityThreshold",vn),D("yt.scheduler.initialized",
!0));function yn(a){var b=new Cj;this.h=(a=b.isAvailable()?a?new Dj(b,a):b:null)?new xj(a):null;this.i=document.domain||window.location.hostname}
yn.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+c*1E3);return}catch(f){}var e="";if(d)try{e=escape((new ni).serialize(b))}catch(f){return}else e=escape(b);ym(a,e,c,this.i)};
yn.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=zm(a))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
yn.prototype.remove=function(a){this.h&&this.h.remove(a);Am(a,"/",this.i)};var zn=function(){var a;return function(){a||(a=new yn("ytidb"));return a}}();
function An(){var a;return(a=zn())==null?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var Bn=[],Cn,Dn=!1;function En(){var a={};for(Cn=new Fn(a.handleError===void 0?Gn:a.handleError,a.logEvent===void 0?Hn:a.logEvent);Bn.length>0;)switch(a=Bn.shift(),a.type){case "ERROR":Cn.Ha(a.payload);break;case "EVENT":Cn.logEvent(a.eventType,a.payload)}}
function In(a){Dn||(Cn?Cn.Ha(a):(Bn.push({type:"ERROR",payload:a}),Bn.length>10&&Bn.shift()))}
function Jn(a,b){Dn||(Cn?Cn.logEvent(a,b):(Bn.push({type:"EVENT",eventType:a,payload:b}),Bn.length>10&&Bn.shift()))}
;function Kn(a){if(a.indexOf(":")>=0)throw Error("Database name cannot contain ':'");}
function Ln(a){return a.substr(0,a.indexOf(":"))||a}
;var Mn=De||Ee;function Nn(a){var b=Jc();return b?b.toLowerCase().indexOf(a)>=0:!1}
;var On={},Pn=(On.AUTH_INVALID="No user identifier specified.",On.EXPLICIT_ABORT="Transaction was explicitly aborted.",On.IDB_NOT_SUPPORTED="IndexedDB is not supported.",On.MISSING_INDEX="Index not created.",On.MISSING_OBJECT_STORES="Object stores not created.",On.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",On.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",On.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
On.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",On.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",On.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",On.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",On),Qn={},Rn=(Qn.AUTH_INVALID="ERROR",Qn.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",Qn.EXPLICIT_ABORT="IGNORED",Qn.IDB_NOT_SUPPORTED="ERROR",Qn.MISSING_INDEX=
"WARNING",Qn.MISSING_OBJECT_STORES="ERROR",Qn.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",Qn.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",Qn.QUOTA_EXCEEDED="WARNING",Qn.QUOTA_MAYBE_EXCEEDED="WARNING",Qn.UNKNOWN_ABORT="WARNING",Qn.INCOMPATIBLE_DB_VERSION="WARNING",Qn),Sn={},Tn=(Sn.AUTH_INVALID=!1,Sn.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,Sn.EXPLICIT_ABORT=!1,Sn.IDB_NOT_SUPPORTED=!1,Sn.MISSING_INDEX=!1,Sn.MISSING_OBJECT_STORES=!1,Sn.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,Sn.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,Sn.QUOTA_EXCEEDED=!1,Sn.QUOTA_MAYBE_EXCEEDED=!0,Sn.UNKNOWN_ABORT=!0,Sn.INCOMPATIBLE_DB_VERSION=!1,Sn);function Un(a,b,c,d,e){b=b===void 0?{}:b;c=c===void 0?Pn[a]:c;d=d===void 0?Rn[a]:d;e=e===void 0?Tn[a]:e;V.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:self.document===void 0,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,Un.prototype)}
y(Un,V);function Vn(a,b){Un.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},Pn.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,Vn.prototype)}
y(Vn,Un);function Wn(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,Wn.prototype)}
y(Wn,Error);var Xn=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function Yn(a,b,c,d){b=Ln(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof Un)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if(e.name==="QuotaExceededError")return new Un("QUOTA_EXCEEDED",a);if(Fe&&e.name==="UnknownError")return new Un("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof Wn)return new Un("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if(e.name==="InvalidStateError"&&Xn.some(function(f){return e.message.includes(f)}))return new Un("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if(e.name==="AbortError")return new Un("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",nd:e.name})];e.level="WARNING";return e}
function Zn(a,b,c){var d=An();return new Un("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:d==null?void 0:d.hasSucceededOnce}})}
;function $n(a){if(!a)throw Error();throw a;}
function ao(a){return a}
function bo(a){this.h=a}
function co(a){function b(e){if(d.state.status==="PENDING"){d.state={status:"REJECTED",reason:e};e=x(d.i);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if(d.state.status==="PENDING"){d.state={status:"FULFILLED",value:e};e=x(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.h=[];this.i=[];a=a.h;try{a(c,b)}catch(e){b(e)}}
co.all=function(a){return new co(new bo(function(b,c){var d=[],e=a.length;e===0&&b(d);for(var f={rb:0};f.rb<a.length;f={rb:f.rb},++f.rb)co.resolve(a[f.rb]).then(function(g){return function(h){d[g.rb]=h;e--;e===0&&b(d)}}(f)).catch(function(g){c(g)})}))};
co.resolve=function(a){return new co(new bo(function(b,c){a instanceof co?a.then(b,c):b(a)}))};
co.reject=function(a){return new co(new bo(function(b,c){c(a)}))};
co.prototype.then=function(a,b){var c=this,d=a!=null?a:ao,e=b!=null?b:$n;return new co(new bo(function(f,g){c.state.status==="PENDING"?(c.h.push(function(){eo(c,c,d,f,g)}),c.i.push(function(){fo(c,c,e,f,g)})):c.state.status==="FULFILLED"?eo(c,c,d,f,g):c.state.status==="REJECTED"&&fo(c,c,e,f,g)}))};
co.prototype.catch=function(a){return this.then(void 0,a)};
function eo(a,b,c,d,e){try{if(a.state.status!=="FULFILLED")throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof co?go(a,b,f,d,e):d(f)}catch(g){e(g)}}
function fo(a,b,c,d,e){try{if(a.state.status!=="REJECTED")throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof co?go(a,b,f,d,e):d(f)}catch(g){e(g)}}
function go(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof co?go(a,b,f,d,e):d(f)},function(f){e(f)})}
;function ho(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function io(a){return new Promise(function(b,c){ho(a,b,c)})}
function jo(a){return new co(new bo(function(b,c){ho(a,b,c)}))}
;function ko(a,b){return new co(new bo(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var lo=window,W=lo.ytcsi&&lo.ytcsi.now?lo.ytcsi.now:lo.performance&&lo.performance.timing&&lo.performance.now&&lo.performance.timing.navigationStart?function(){return lo.performance.timing.navigationStart+lo.performance.now()}:function(){return(new Date).getTime()};function mo(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(W());this.i=!1}
n=mo.prototype;n.add=function(a,b,c){return no(this,[a],{mode:"readwrite",la:!0},function(d){return d.objectStore(a).add(b,c)})};
n.clear=function(a){return no(this,[a],{mode:"readwrite",la:!0},function(b){return b.objectStore(a).clear()})};
n.close=function(){this.h.close();var a;((a=this.options)==null?0:a.closed)&&this.options.closed()};
n.count=function(a,b){return no(this,[a],{mode:"readonly",la:!0},function(c){return c.objectStore(a).count(b)})};
function oo(a,b,c){a=a.h.createObjectStore(b,c);return new po(a)}
n.delete=function(a,b){return no(this,[a],{mode:"readwrite",la:!0},function(c){return c.objectStore(a).delete(b)})};
n.get=function(a,b){return no(this,[a],{mode:"readonly",la:!0},function(c){return c.objectStore(a).get(b)})};
function qo(a,b,c){return no(a,[b],{mode:"readwrite",la:!0},function(d){d=d.objectStore(b);return jo(d.h.put(c,void 0))})}
n.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function no(a,b,c,d){var e,f,g,h,k,l,m,p,t,r,v,u;return A(function(z){switch(z.h){case 1:var F={mode:"readonly",la:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};typeof c==="string"?F.mode=c:Object.assign(F,c);e=F;a.transactionCount++;f=e.la?3:1;g=0;case 2:if(h){z.B(4);break}g++;k=Math.round(W());Aa(z,5);l=a.h.transaction(b,e.mode);F=z.yield;var J=new ro(l);J=so(J,d);return F.call(z,J,7);case 7:return m=z.i,p=Math.round(W()),to(a,k,p,g,void 0,b.join(),e),z.return(m);case 5:t=Ba(z);r=Math.round(W());v=Yn(t,
a.h.name,b.join(),a.h.version);if((u=v instanceof Un&&!v.h)||g>=f)to(a,k,r,g,v,b.join(),e),h=v;z.B(2);break;case 4:return z.return(Promise.reject(h))}})}
function to(a,b,c,d,e,f,g){b=c-b;e?(e instanceof Un&&(e.type==="QUOTA_EXCEEDED"||e.type==="QUOTA_MAYBE_EXCEEDED")&&Jn("QUOTA_EXCEEDED",{dbName:Ln(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof Un&&e.type==="UNKNOWN_ABORT"&&(c-=a.j,c<0&&c>=Math.pow(2,31)&&(c=0),Jn("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.i=!0),uo(a,!1,d,f,b,g.tag),In(e)):uo(a,!0,d,f,b,g.tag)}
function uo(a,b,c,d,e,f){Jn("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c,tag:f===void 0?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
n.getName=function(){return this.h.name};
function po(a){this.h=a}
n=po.prototype;n.add=function(a,b){return jo(this.h.add(a,b))};
n.autoIncrement=function(){return this.h.autoIncrement};
n.clear=function(){return jo(this.h.clear()).then(function(){})};
function vo(a,b,c){a.h.createIndex(b,c,{unique:!1})}
n.count=function(a){return jo(this.h.count(a))};
function wo(a,b){return xo(a,{query:b},function(c){return c.delete().then(function(){return yo(c)})}).then(function(){})}
n.delete=function(a){return a instanceof IDBKeyRange?wo(this,a):jo(this.h.delete(a))};
n.get=function(a){return jo(this.h.get(a))};
n.index=function(a){try{return new zo(this.h.index(a))}catch(b){if(b instanceof Error&&b.name==="NotFoundError")throw new Wn(a,this.h.name);throw b;}};
n.getName=function(){return this.h.name};
n.keyPath=function(){return this.h.keyPath};
function xo(a,b,c){a=a.h.openCursor(b.query,b.direction);return Ao(a).then(function(d){return ko(d,c)})}
function ro(a){var b=this;this.h=a;this.i=new Map;this.aborted=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.aborted){e=Un;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(k===null)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function so(a,b){var c=new Promise(function(d,e){try{b(a).then(function(f){d(f)}).catch(e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return x(d).next().value})}
ro.prototype.abort=function(){this.h.abort();this.aborted=!0;throw new Un("EXPLICIT_ABORT");};
ro.prototype.objectStore=function(a){a=this.h.objectStore(a);var b=this.i.get(a);b||(b=new po(a),this.i.set(a,b));return b};
function zo(a){this.h=a}
n=zo.prototype;n.count=function(a){return jo(this.h.count(a))};
n.delete=function(a){return Bo(this,{query:a},function(b){return b.delete().then(function(){return yo(b)})})};
n.get=function(a){return jo(this.h.get(a))};
n.keyPath=function(){return this.h.keyPath};
n.unique=function(){return this.h.unique};
function Bo(a,b,c){a=a.h.openCursor(b.query===void 0?null:b.query,b.direction===void 0?"next":b.direction);return Ao(a).then(function(d){return ko(d,c)})}
function Co(a,b){this.request=a;this.cursor=b}
function Ao(a){return jo(a).then(function(b){return b?new Co(a,b):null})}
function yo(a){a.cursor.continue(void 0);return Ao(a.request)}
Co.prototype.delete=function(){return jo(this.cursor.delete()).then(function(){})};
Co.prototype.getValue=function(){return this.cursor.value};
Co.prototype.update=function(a){return jo(this.cursor.update(a))};function Do(a,b,c){return new Promise(function(d,e){function f(){t||(t=new mo(g.result,{closed:p}));return t}
var g=b!==void 0?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.Nd,k=c.blocking,l=c.ef,m=c.upgrade,p=c.closed,t;g.addEventListener("upgradeneeded",function(r){try{if(r.newVersion===null)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(g.transaction===null)throw Error("Invariant: transaction on IDbOpenDbRequest is null");r.dataLoss&&r.dataLoss!=="none"&&Jn("IDB_DATA_CORRUPTED",{reason:r.dataLossMessage||"unknown reason",dbName:Ln(a)});var v=f(),u=new ro(g.transaction);
m&&m(v,function(z){return r.oldVersion<z&&r.newVersion>=z},u);
u.done.catch(function(z){e(z)})}catch(z){e(z)}});
g.addEventListener("success",function(){var r=g.result;k&&r.addEventListener("versionchange",function(){k(f())});
r.addEventListener("close",function(){Jn("IDB_UNEXPECTEDLY_CLOSED",{dbName:Ln(a),dbVersion:r.version});l&&l()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function Eo(a,b,c){c=c===void 0?{}:c;return Do(a,b,c)}
function Fo(a,b){b=b===void 0?{}:b;var c,d,e,f;return A(function(g){if(g.h==1)return Aa(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.Nd)&&c.addEventListener("blocked",function(){e()}),g.yield(io(c),4);
if(g.h!=2)g.h=0,g.l=0;else throw f=Ba(g),Yn(f,a,"",-1);})}
;function Go(a,b){this.name=a;this.options=b;this.j=!0;this.v=this.l=0}
Go.prototype.i=function(a,b,c){c=c===void 0?{}:c;return Eo(a,b,c)};
Go.prototype.delete=function(a){a=a===void 0?{}:a;return Fo(this.name,a)};
function Ho(a,b){return new Un("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function Io(a,b){if(!b)throw Zn("openWithToken",Ln(a.name));return a.open()}
Go.prototype.open=function(){function a(){var f,g,h,k,l,m,p,t,r,v;return A(function(u){switch(u.h){case 1:return g=(f=Error().stack)!=null?f:"",Aa(u,2),u.yield(c.i(c.name,c.options.version,e),4);case 4:for(var z=h=u.i,F=c.options,J=[],O=x(Object.keys(F.xb)),S=O.next();!S.done;S=O.next()){S=S.value;var da=F.xb[S],va=da.Ge===void 0?Number.MAX_VALUE:da.Ge;!(z.h.version>=da.Fb)||z.h.version>=va||z.h.objectStoreNames.contains(S)||J.push(S)}k=J;if(k.length===0){u.B(5);break}l=Object.keys(c.options.xb);
m=h.objectStoreNames();if(c.v<Ol("ytidb_reopen_db_retries",0))return c.v++,h.close(),In(new Un("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),u.return(a());if(!(c.l<Ol("ytidb_remake_db_retries",1))){u.B(6);break}c.l++;return u.yield(c.delete(),7);case 7:return In(new Un("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),u.return(a());case 6:throw new Vn(m,l);case 5:return u.return(h);case 2:p=Ba(u);
if(p instanceof DOMException?p.name!=="VersionError":"DOMError"in self&&p instanceof DOMError?p.name!=="VersionError":!(p instanceof Object&&"message"in p)||p.message!=="An attempt was made to open a database using a lower version than the existing version."){u.B(8);break}return u.yield(c.i(c.name,void 0,Object.assign({},e,{upgrade:void 0})),9);case 9:t=u.i;r=t.h.version;if(c.options.version!==void 0&&r>c.options.version+1)throw t.close(),c.j=!1,Ho(c,r);return u.return(t);case 8:throw b(),p instanceof
Error&&!U("ytidb_async_stack_killswitch")&&(p.stack=p.stack+"\n"+g.substring(g.indexOf("\n")+1)),Yn(p,c.name,"",(v=c.options.version)!=null?v:-1);}})}
function b(){c.h===d&&(c.h=void 0)}
var c=this;if(!this.j)throw Ho(this);if(this.h)return this.h;var d,e={blocking:function(f){f.close()},
closed:b,ef:b,upgrade:this.options.upgrade};return this.h=d=a()};var Jo=new Go("YtIdbMeta",{xb:{databases:{Fb:1}},upgrade:function(a,b){b(1)&&oo(a,"databases",{keyPath:"actualName"})}});
function Ko(a,b){var c;return A(function(d){if(d.h==1)return d.yield(Io(Jo,b),2);c=d.i;return d.return(no(c,["databases"],{la:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return jo(f.h.put(a,void 0)).then(function(){})})}))})}
function Lo(a,b){var c;return A(function(d){if(d.h==1)return a?d.yield(Io(Jo,b),2):d.return();c=d.i;return d.return(c.delete("databases",a))})}
function Mo(a,b){var c,d;return A(function(e){return e.h==1?(c=[],e.yield(Io(Jo,b),2)):e.h!=3?(d=e.i,e.yield(no(d,["databases"],{la:!0,mode:"readonly"},function(f){c.length=0;return xo(f.objectStore("databases"),{},function(g){a(g.getValue())&&c.push(g.getValue());return yo(g)})}),3)):e.return(c)})}
function No(a){return Mo(function(b){return b.publicName==="LogsDatabaseV2"&&b.userIdentifier!==void 0},a)}
function Oo(a,b,c){return Mo(function(d){return c?d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)&&c.includes(d.publicName):d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)},b)}
function Po(a){var b,c;return A(function(d){if(d.h==1)return b=Qm("YtIdbMeta hasAnyMeta other"),d.yield(Mo(function(e){return e.userIdentifier!==void 0&&e.userIdentifier!==b},a),2);
c=d.i;return d.return(c.length>0)})}
;var Qo,Ro=new function(){}(new function(){});
function So(){var a,b,c,d;return A(function(e){switch(e.h){case 1:a=An();if((b=a)==null?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=Mn)f=/WebKit\/([0-9]+)/.exec(Jc()),f=!!(f&&parseInt(f[1],10)>=600);f&&(f=/WebKit\/([0-9]+)/.exec(Jc()),f=!(f&&parseInt(f[1],10)>=602));if(f||Wc)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
Aa(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return e.yield(Ko(d,Ro),4);case 4:return e.yield(Lo("yt-idb-test-do-not-use",Ro),5);case 5:return e.return(!0);case 2:return Ba(e),e.return(!1)}})}
function To(){if(Qo!==void 0)return Qo;Dn=!0;return Qo=So().then(function(a){Dn=!1;var b;if((b=zn())!=null&&b.h){var c;b={hasSucceededOnce:((c=An())==null?void 0:c.hasSucceededOnce)||a};var d;(d=zn())==null||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function Uo(){return E("ytglobal.idbToken_")||void 0}
function Vo(){var a=Uo();return a?Promise.resolve(a):To().then(function(b){(b=b?Ro:void 0)&&D("ytglobal.idbToken_",b);return b})}
;var Wo=0;function Xo(a,b){Wo||(Wo=cj.pa(function(){var c,d,e,f,g;return A(function(h){switch(h.h){case 1:return h.yield(Vo(),2);case 2:c=h.i;if(!c)return h.return();d=!0;Aa(h,3);return h.yield(Oo(a,c,b),5);case 5:e=h.i;if(!e.length){d=!1;h.B(6);break}f=e[0];return h.yield(Fo(f.actualName),7);case 7:return h.yield(Lo(f.actualName,c),6);case 6:h.h=4;h.l=0;break;case 3:g=Ba(h),In(g),d=!1;case 4:cj.qa(Wo),Wo=0,d&&Xo(a,b),h.h=0}})}))}
function Yo(){var a;return A(function(b){return b.h==1?b.yield(Vo(),2):(a=b.i)?b.return(Po(a)):b.return(!1)})}
new Mi;function Zo(a){if(!Pm())throw a=new Un("AUTH_INVALID",{dbName:a}),In(a),a;var b=Qm();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function $o(a,b,c,d){var e,f,g,h,k,l;return A(function(m){switch(m.h){case 1:return f=(e=Error().stack)!=null?e:"",m.yield(Vo(),2);case 2:g=m.i;if(!g)throw h=Zn("openDbImpl",a,b),U("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),In(h),h;Kn(a);k=c?{actualName:a,publicName:a,userIdentifier:void 0}:Zo(a);Aa(m,3);return m.yield(Ko(k,g),5);case 5:return m.yield(Eo(k.actualName,b,d),6);case 6:return m.return(m.i);case 3:return l=Ba(m),Aa(m,7),m.yield(Lo(k.actualName,
g),9);case 9:m.h=8;m.l=0;break;case 7:Ba(m);case 8:throw l;}})}
function ap(a,b,c){c=c===void 0?{}:c;return $o(a,b,!1,c)}
function bp(a,b,c){c=c===void 0?{}:c;return $o(a,b,!0,c)}
function cp(a,b){b=b===void 0?{}:b;var c,d;return A(function(e){if(e.h==1)return e.yield(Vo(),2);if(e.h!=3){c=e.i;if(!c)return e.return();Kn(a);d=Zo(a);return e.yield(Fo(d.actualName,b),3)}return e.yield(Lo(d.actualName,c),0)})}
function dp(a,b,c){a=a.map(function(d){return A(function(e){return e.h==1?e.yield(Fo(d.actualName,b),2):e.yield(Lo(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function ep(){var a=a===void 0?{}:a;var b,c;return A(function(d){if(d.h==1)return d.yield(Vo(),2);if(d.h!=3){b=d.i;if(!b)return d.return();Kn("LogsDatabaseV2");return d.yield(No(b),3)}c=d.i;return d.yield(dp(c,a,b),0)})}
function fp(a,b){b=b===void 0?{}:b;var c;return A(function(d){if(d.h==1)return d.yield(Vo(),2);if(d.h!=3){c=d.i;if(!c)return d.return();Kn(a);return d.yield(Fo(a,b),3)}return d.yield(Lo(a,c),0)})}
;function gp(a,b){Go.call(this,a,b);this.options=b;Kn(a)}
y(gp,Go);function hp(a,b){var c;return function(){c||(c=new gp(a,b));return c}}
gp.prototype.i=function(a,b,c){c=c===void 0?{}:c;return(this.options.shared?bp:ap)(a,b,Object.assign({},c))};
gp.prototype.delete=function(a){a=a===void 0?{}:a;return(this.options.shared?fp:cp)(this.name,a)};
function ip(a,b){return hp(a,b)}
;var jp={},kp=ip("ytGcfConfig",{xb:(jp.coldConfigStore={Fb:1},jp.hotConfigStore={Fb:1},jp),shared:!1,upgrade:function(a,b){b(1)&&(vo(oo(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),vo(oo(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function lp(a){return Io(kp(),a)}
function mp(a,b,c){var d,e,f;return A(function(g){switch(g.h){case 1:return d={config:a,hashData:b,timestamp:W()},g.yield(lp(c),2);case 2:return e=g.i,g.yield(e.clear("hotConfigStore"),3);case 3:return g.yield(qo(e,"hotConfigStore",d),4);case 4:return f=g.i,g.return(f)}})}
function np(a,b,c,d){var e,f,g;return A(function(h){switch(h.h){case 1:return e={config:a,hashData:b,configData:c,timestamp:W()},h.yield(lp(d),2);case 2:return f=h.i,h.yield(f.clear("coldConfigStore"),3);case 3:return h.yield(qo(f,"coldConfigStore",e),4);case 4:return g=h.i,h.return(g)}})}
function op(a){var b,c;return A(function(d){return d.h==1?d.yield(lp(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(no(b,["coldConfigStore"],{mode:"readwrite",la:!0},function(e){return Bo(e.objectStore("coldConfigStore").index("coldTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
function pp(a){var b,c;return A(function(d){return d.h==1?d.yield(lp(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(no(b,["hotConfigStore"],{mode:"readwrite",la:!0},function(e){return Bo(e.objectStore("hotConfigStore").index("hotTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
;function qp(){G.call(this);this.i=[];this.h=[];var a=E("yt.gcf.config.hotUpdateCallbacks");a?(this.i=[].concat(la(a)),this.h=a):(this.h=[],D("yt.gcf.config.hotUpdateCallbacks",this.h))}
y(qp,G);qp.prototype.U=function(){for(var a=x(this.i),b=a.next();!b.done;b=a.next()){var c=this.h;b=c.indexOf(b.value);b>=0&&c.splice(b,1)}this.i.length=0;G.prototype.U.call(this)};function rp(){this.h=0;this.i=new qp}
function sp(){var a;return(a=E("yt.gcf.config.hotConfigGroup"))!=null?a:T("RAW_HOT_CONFIG_GROUP")}
function tp(a,b,c){var d,e,f;return A(function(g){switch(g.h){case 1:if(!U("start_client_gcf")){g.B(0);break}c&&(a.j=c,D("yt.gcf.config.hotConfigGroup",a.j||null));a.l(b);d=Uo();if(!d){g.B(3);break}if(c){g.B(4);break}return g.yield(pp(d),5);case 5:e=g.i,c=(f=e)==null?void 0:f.config;case 4:return g.yield(mp(c,b,d),3);case 3:if(c)for(var h=c,k=x(a.i.h),l=k.next();!l.done;l=k.next())l=l.value,l(h);g.h=0}})}
function up(a,b,c){var d,e,f,g;return A(function(h){if(h.h==1){if(!U("start_client_gcf"))return h.B(0);a.coldHashData=b;D("yt.gcf.config.coldHashData",a.coldHashData||null);return(d=Uo())?c?h.B(4):h.yield(op(d),5):h.B(0)}h.h!=4&&(e=h.i,c=(f=e)==null?void 0:f.config);if(!c)return h.B(0);g=c.configData;return h.yield(np(c,b,g,d),0)})}
function vp(){if(!rp.h){var a=new rp;rp.h=a}a=rp.h;var b=W()-a.h;if(!(a.h!==0&&b<Ol("send_config_hash_timer"))){b=E("yt.gcf.config.coldConfigData");var c=E("yt.gcf.config.hotHashData"),d=E("yt.gcf.config.coldHashData");b&&c&&d&&(a.h=W());return{coldConfigData:b,hotHashData:c,coldHashData:d}}}
rp.prototype.l=function(a){this.hotHashData=a;D("yt.gcf.config.hotHashData",this.hotHashData||null)};function wp(){return"INNERTUBE_API_KEY"in ol&&"INNERTUBE_API_VERSION"in ol}
function xp(){return{innertubeApiKey:T("INNERTUBE_API_KEY"),innertubeApiVersion:T("INNERTUBE_API_VERSION"),je:T("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),hd:T("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),Kg:T("INNERTUBE_CONTEXT_CLIENT_NAME",1),innertubeContextClientVersion:T("INNERTUBE_CONTEXT_CLIENT_VERSION"),le:T("INNERTUBE_CONTEXT_HL"),ke:T("INNERTUBE_CONTEXT_GL"),me:T("INNERTUBE_HOST_OVERRIDE")||"",oe:!!T("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),ne:!!T("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:T("SERIALIZED_CLIENT_CONFIG_DATA")}}
function yp(a){var b={client:{hl:a.le,gl:a.ke,clientName:a.hd,clientVersion:a.innertubeContextClientVersion,configInfo:a.je}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=C.devicePixelRatio;c&&c!=1&&(b.client.screenDensityFloat=String(c));c=T("EXPERIMENTS_TOKEN","");c!==""&&(b.client.experimentsToken=c);c=Pl();c.length>0&&(b.request={internalExperimentFlags:c});c=a.hd;if((c==="WEB"||c==="MWEB"||c===1||c===2)&&b){var d;b.client.mainAppWebInfo=(d=b.client.mainAppWebInfo)!=
null?d:{};b.client.mainAppWebInfo.webDisplayMode=sm()}(d=E("yt.embedded_player.embed_url"))&&b&&(b.thirdParty={embedUrl:d});var e;if(U("web_log_memory_total_kbytes")&&((e=C.navigator)==null?0:e.deviceMemory)){var f;e=(f=C.navigator)==null?void 0:f.deviceMemory;b&&(b.client.memoryTotalKbytes=""+e*1E6)}a.appInstallData&&b&&(b.client.configInfo=b.client.configInfo||{},b.client.configInfo.appInstallData=a.appInstallData);(a=Nm())&&b&&(b.client.connectionType=a);U("web_log_effective_connection_type")&&
(a=Om())&&b&&(b.client.effectiveConnectionType=a);U("start_client_gcf")&&(e=vp())&&(a=e.coldConfigData,f=e.coldHashData,e=e.hotHashData,b&&(b.client.configInfo=b.client.configInfo||{},a&&(b.client.configInfo.coldConfigData=a),f&&(b.client.configInfo.coldHashData=f),e&&(b.client.configInfo.hotHashData=e)));T("DELEGATED_SESSION_ID")&&!U("pageid_as_header_web")&&(b.user={onBehalfOfUser:T("DELEGATED_SESSION_ID")});!U("fill_delegate_context_in_gel_killswitch")&&(a=T("INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT"))&&
(b.user=Object.assign({},b.user,{serializedDelegationContext:a}));a=Object;f=a.assign;e=b.client;d={};c=x(Object.entries(Cl(T("DEVICE",""))));for(var g=c.next();!g.done;g=c.next()){var h=x(g.value);g=h.next().value;h=h.next().value;g==="cbrand"?d.deviceMake=h:g==="cmodel"?d.deviceModel=h:g==="cbr"?d.browserName=h:g==="cbrver"?d.browserVersion=h:g==="cos"?d.osName=h:g==="cosver"?d.osVersion=h:g==="cplatform"&&(d.platform=h)}b.client=f.call(a,e,d);return b}
function zp(a,b,c){c=c===void 0?{}:c;var d={};T("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":T("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||T("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;b=c.authorization||T("AUTHORIZATION");b||(a?b="Bearer "+E("gapi.auth.getToken")().Cg:(a=wm(um()),U("pageid_as_header_web")||delete a["X-Goog-PageId"],d=Object.assign({},d,a)));b&&(d.Authorization=b);return d}
;var Ap=typeof TextEncoder!=="undefined"?new TextEncoder:null,Bp=Ap?function(a){return Ap.encode(a)}:function(a){for(var b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);
e<128?b[c++]=e:(e<2048?b[c++]=e>>6|192:((e&64512)==55296&&d+1<a.length&&(a.charCodeAt(d+1)&64512)==56320?(e=65536+((e&1023)<<10)+(a.charCodeAt(++d)&1023),b[c++]=e>>18|240,b[c++]=e>>12&63|128):b[c++]=e>>12|224,b[c++]=e>>6&63|128),b[c++]=e&63|128)}a=new Uint8Array(b.length);for(c=0;c<a.length;c++)a[c]=b[c];return a};function Cp(a,b){this.version=a;this.args=b}
Cp.prototype.serialize=function(){return{version:this.version,args:this.args}};function Dp(a,b){this.topic=a;this.h=b}
Dp.prototype.toString=function(){return this.topic};var Ep=E("ytPubsub2Pubsub2Instance")||new M;M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.Bb;M.prototype.publish=M.prototype.Ya;M.prototype.clear=M.prototype.clear;D("ytPubsub2Pubsub2Instance",Ep);var Fp=E("ytPubsub2Pubsub2SubscribedKeys")||{};D("ytPubsub2Pubsub2SubscribedKeys",Fp);var Gp=E("ytPubsub2Pubsub2TopicToKeys")||{};D("ytPubsub2Pubsub2TopicToKeys",Gp);var Hp=E("ytPubsub2Pubsub2IsAsync")||{};D("ytPubsub2Pubsub2IsAsync",Hp);
D("ytPubsub2Pubsub2SkipSubKey",null);function Ip(a,b){var c=Jp();c&&c.publish.call(c,a.toString(),a,b)}
function Kp(a){var b=Lp,c=Jp();if(!c)return 0;var d=c.subscribe(b.toString(),function(e,f){var g=E("ytPubsub2Pubsub2SkipSubKey");g&&g==d||(g=function(){if(Fp[d])try{if(f&&b instanceof Dp&&b!=e)try{var h=b.h,k=f;if(!k.args||!k.version)throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");try{if(!h.Dd){var l=new h;h.Dd=l.version}var m=h.Dd}catch(z){}if(!m||k.version!=m)throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");try{m=Reflect;var p=m.construct;
var t=k.args,r=t.length;if(r>0){var v=Array(r);for(k=0;k<r;k++)v[k]=t[k];var u=v}else u=[];f=p.call(m,h,u)}catch(z){throw z.message="yt.pubsub2.Data.deserialize(): "+z.message,z;}}catch(z){throw z.message="yt.pubsub2.pubsub2 cross-binary conversion error for "+b.toString()+": "+z.message,z;}a.call(window,f)}catch(z){ul(z)}},Hp[b.toString()]?E("yt.scheduler.instance")?cj.pa(g):Ll(g,0):g())});
Fp[d]=!0;Gp[b.toString()]||(Gp[b.toString()]=[]);Gp[b.toString()].push(d);return d}
function Mp(){var a=Np,b=Kp(function(c){a.apply(void 0,arguments);Op(b)});
return b}
function Op(a){var b=Jp();b&&(typeof a==="number"&&(a=[a]),Db(a,function(c){b.unsubscribeByKey(c);delete Fp[c]}))}
function Jp(){return E("ytPubsub2Pubsub2Instance")}
;function Pp(a,b,c){c=c===void 0?{sampleRate:.1}:c;Math.random()<Math.min(.02,c.sampleRate/100)&&Ip("meta_logging_csi_event",{timerName:a,ah:b})}
;var Qp=void 0,Rp=void 0;function Sp(){Rp||(Rp=Rk(T("WORKER_SERIALIZATION_URL")));return Rp||void 0}
function Tp(){var a=Sp();Qp||a===void 0||(Qp=new Worker(kb(a),void 0));return Qp}
;var Up=Ol("max_body_size_to_compress",5E5),Vp=Ol("min_body_size_to_compress",500),Wp=!0,Xp=0,Yp=0,Zp=Ol("compression_performance_threshold_lr",250),$p=Ol("slow_compressions_before_abandon_count",4),aq=!1,bq=new Map,cq=1,dq=!0;function eq(){if(typeof Worker==="function"&&Sp()&&!aq){var a=function(c){c=c.data;if(c.op==="gzippedGelBatch"){var d=bq.get(c.key);d&&(fq(c.gzippedBatch,d.latencyPayload,d.url,d.options,d.sendFn),bq.delete(c.key))}},b=Tp();
b&&(b.addEventListener("message",a),b.onerror=function(){bq.clear()},aq=!0)}}
function gq(a,b,c,d,e){e=e===void 0?!1:e;var f={startTime:W(),ticks:{},infos:{}};if(Wp)try{var g=hq(b);if(g!=null&&(g>Up||g<Vp))d(a,c);else{if(U("gzip_gel_with_worker")&&(U("initial_gzip_use_main_thread")&&!dq||!U("initial_gzip_use_main_thread"))){aq||eq();var h=Tp();if(h&&!e){bq.set(cq,{latencyPayload:f,url:a,options:c,sendFn:d});h.postMessage({op:"gelBatchToGzip",serializedBatch:b,key:cq});cq++;return}}var k=Qk(Bp(b));fq(k,f,a,c,d)}}catch(l){vl(l),d(a,c)}else d(a,c)}
function fq(a,b,c,d,e){dq=!1;var f=W();b.ticks.gelc=f;Yp++;U("disable_compression_due_to_performance_degredation")&&f-b.startTime>=Zp&&(Xp++,U("abandon_compression_after_N_slow_zips")?Yp===Ol("compression_disable_point")&&Xp>$p&&(Wp=!1):Wp=!1);iq(b);d.headers||(d.headers={});d.headers["Content-Encoding"]="gzip";d.postBody=a;d.postParams=void 0;e(c,d)}
function jq(a){var b=b===void 0?!1:b;var c=c===void 0?!1:c;var d=W(),e={startTime:d,ticks:{},infos:{}},f=b?E("yt.logging.gzipForFetch",!1):!0;if(Wp&&f){if(!a.body)return a;try{var g=c?a.body:typeof a.body==="string"?a.body:JSON.stringify(a.body);f=g;if(!c&&typeof g==="string"){var h=hq(g);if(h!=null&&(h>Up||h<Vp))return a;c=b?{level:1}:void 0;f=Qk(Bp(g),c);var k=W();e.ticks.gelc=k;if(b){Yp++;if((U("disable_compression_due_to_performance_degredation")||U("disable_compression_due_to_performance_degradation_lr"))&&
k-d>=Zp)if(Xp++,U("abandon_compression_after_N_slow_zips")||U("abandon_compression_after_N_slow_zips_lr")){b=Xp/Yp;var l=$p/Ol("compression_disable_point");Yp>0&&Yp%Ol("compression_disable_point")===0&&b>=l&&(Wp=!1)}else Wp=!1;iq(e)}}a.headers=Object.assign({},{"Content-Encoding":"gzip"},a.headers||{});a.body=f;return a}catch(m){return vl(m),a}}else return a}
function hq(a){try{return(new Blob(a.split(""))).size}catch(b){return vl(b),null}}
function iq(a){U("gel_compression_csi_killswitch")||!U("log_gel_compression_latency")&&!U("log_gel_compression_latency_lr")||Pp("gel_compression",a,{sampleRate:.1})}
;function kq(a){a=Object.assign({},a);delete a.Authorization;var b=Ch();if(b){var c=new gj;c.update(T("INNERTUBE_API_KEY"));c.update(b);a.hash=Ie(c.digest(),3)}return a}
;var lq;function mq(){lq||(lq=new yn("yt.innertube"));return lq}
function nq(a,b,c,d){if(d)return null;d=mq().get("nextId",!0)||1;var e=mq().get("requests",!0)||{};e[d]={method:a,request:b,authState:kq(c),requestTime:Math.round(W())};mq().set("nextId",d+1,86400,!0);mq().set("requests",e,86400,!0);return d}
function oq(a){var b=mq().get("requests",!0)||{};delete b[a];mq().set("requests",b,86400,!0)}
function pq(a){var b=mq().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(Math.round(W())-d.requestTime<6E4)){var e=d.authState,f=kq(zp(!1));Qb(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(W())),qq(a,d.method,e,{}));delete b[c]}}mq().set("requests",b,86400,!0)}}
;function rq(a){this.Xb=this.h=!1;this.potentialEsfErrorCounter=this.i=0;this.handleError=function(){};
this.pb=function(){};
this.now=Date.now;this.Ib=!1;var b;this.Ad=(b=a.Ad)!=null?b:100;var c;this.ud=(c=a.ud)!=null?c:1;var d;this.rd=(d=a.rd)!=null?d:2592E6;var e;this.pd=(e=a.pd)!=null?e:12E4;var f;this.td=(f=a.td)!=null?f:5E3;var g;this.X=(g=a.X)!=null?g:void 0;this.dc=!!a.dc;var h;this.ac=(h=a.ac)!=null?h:.1;var k;this.mc=(k=a.mc)!=null?k:10;a.handleError&&(this.handleError=a.handleError);a.pb&&(this.pb=a.pb);a.Ib&&(this.Ib=a.Ib);a.Xb&&(this.Xb=a.Xb);this.Y=a.Y;this.Da=a.Da;this.ha=a.ha;this.fa=a.fa;this.sendFn=a.sendFn;
this.Kc=a.Kc;this.Hc=a.Hc;sq(this)&&(!this.Y||this.Y("networkless_logging"))&&tq(this)}
function tq(a){sq(a)&&!a.Ib&&(a.h=!0,a.dc&&Math.random()<=a.ac&&a.ha.Pd(a.X),uq(a),a.fa.va()&&a.Sb(),a.fa.listen(a.Kc,a.Sb.bind(a)),a.fa.listen(a.Hc,a.Vc.bind(a)))}
n=rq.prototype;n.writeThenSend=function(a,b){var c=this;b=b===void 0?{}:b;if(sq(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.ha.set(d,this.X).then(function(e){d.id=e;c.fa.va()&&vq(c,d)}).catch(function(e){vq(c,d);
wq(c,e)})}else this.sendFn(a,b)};
n.sendThenWrite=function(a,b,c){var d=this;b=b===void 0?{}:b;if(sq(this)&&this.h){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.Y&&this.Y("nwl_skip_retry")&&(e.skipRetry=c);if(this.fa.va()||this.Y&&this.Y("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return A(function(k){if(k.h==1)return k.yield(d.ha.set(e,d.X).catch(function(l){wq(d,l)}),2);
f(g,h);k.h=0})}}this.sendFn(a,b,e.skipRetry)}else this.ha.set(e,this.X).catch(function(g){d.sendFn(a,b,e.skipRetry);
wq(d,g)})}else this.sendFn(a,b,this.Y&&this.Y("nwl_skip_retry")&&c)};
n.sendAndWrite=function(a,b){var c=this;b=b===void 0?{}:b;if(sq(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){d.id!==void 0?c.ha.ob(d.id,c.X):e=!0;c.fa.gb&&c.Y&&c.Y("vss_network_hint")&&c.fa.gb(!0);f(g,h)};
this.sendFn(d.url,d.options,void 0,!0);this.ha.set(d,this.X).then(function(g){d.id=g;e&&c.ha.ob(d.id,c.X)}).catch(function(g){wq(c,g)})}else this.sendFn(a,b,void 0,!0)};
n.Sb=function(){var a=this;if(!sq(this))throw Error("IndexedDB is not supported: throttleSend");this.i||(this.i=this.Da.pa(function(){var b;return A(function(c){if(c.h==1)return c.yield(a.ha.dd("NEW",a.X),2);if(c.h!=3)return b=c.i,b?c.yield(vq(a,b),3):(a.Vc(),c.return());a.i&&(a.i=0,a.Sb());c.h=0})},this.Ad))};
n.Vc=function(){this.Da.qa(this.i);this.i=0};
function vq(a,b){var c;return A(function(d){switch(d.h){case 1:if(!sq(a))throw Error("IndexedDB is not supported: immediateSend");if(b.id===void 0){d.B(2);break}return d.yield(a.ha.se(b.id,a.X),3);case 3:(c=d.i)||a.pb(Error("The request cannot be found in the database."));case 2:if(xq(a,b,a.rd)){d.B(4);break}a.pb(Error("Networkless Logging: Stored logs request expired age limit"));if(b.id===void 0){d.B(5);break}return d.yield(a.ha.ob(b.id,a.X),5);case 5:return d.return();case 4:b.skipRetry||(b=yq(a,
b));if(!b){d.B(0);break}if(!b.skipRetry||b.id===void 0){d.B(8);break}return d.yield(a.ha.ob(b.id,a.X),8);case 8:a.sendFn(b.url,b.options,!!b.skipRetry),d.h=0}})}
function yq(a,b){if(!sq(a))throw Error("IndexedDB is not supported: updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,k,l;return A(function(m){switch(m.h){case 1:g=zq(f);(h=Aq(f))&&a.Y&&a.Y("web_enable_error_204")&&a.handleError(Error("Request failed due to compression"),b.url,f);if(!(a.Y&&a.Y("nwl_consider_error_code")&&g||a.Y&&!a.Y("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.mc)){m.B(2);break}if(!a.fa.pc){m.B(3);break}return m.yield(a.fa.pc(),3);case 3:if(a.fa.va()){m.B(2);break}c(e,f);if(!a.Y||!a.Y("nwl_consider_error_code")||((k=b)==null?void 0:k.id)===void 0){m.B(6);
break}return m.yield(a.ha.Lc(b.id,a.X,!1),6);case 6:return m.return();case 2:if(a.Y&&a.Y("nwl_consider_error_code")&&!g&&a.potentialEsfErrorCounter>a.mc)return m.return();a.potentialEsfErrorCounter++;if(((l=b)==null?void 0:l.id)===void 0){m.B(8);break}return b.sendCount<a.ud?m.yield(a.ha.Lc(b.id,a.X,!0,h?!1:void 0),12):m.yield(a.ha.ob(b.id,a.X),8);case 12:a.Da.pa(function(){a.fa.va()&&a.Sb()},a.td);
case 8:c(e,f),m.h=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return A(function(h){if(h.h==1)return((g=b)==null?void 0:g.id)===void 0?h.B(2):h.yield(a.ha.ob(b.id,a.X),2);a.fa.gb&&a.Y&&a.Y("vss_network_hint")&&a.fa.gb(!0);d(e,f);h.h=0})};
return b}
function xq(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function uq(a){if(!sq(a))throw Error("IndexedDB is not supported: retryQueuedRequests");a.ha.dd("QUEUED",a.X).then(function(b){b&&!xq(a,b,a.pd)?a.Da.pa(function(){return A(function(c){if(c.h==1)return b.id===void 0?c.B(2):c.yield(a.ha.Lc(b.id,a.X),2);uq(a);c.h=0})}):a.fa.va()&&a.Sb()})}
function wq(a,b){a.Gd&&!a.fa.va()?a.Gd(b):a.handleError(b)}
function sq(a){return!!a.X||a.Xb}
function zq(a){var b;return(a=a==null?void 0:(b=a.error)==null?void 0:b.code)&&a>=400&&a<=599?!1:!0}
function Aq(a){var b;a=a==null?void 0:(b=a.error)==null?void 0:b.code;return!(a!==400&&a!==415)}
;var Bq;
function Cq(){if(Bq)return Bq();var a={};Bq=ip("LogsDatabaseV2",{xb:(a.LogsRequestsStore={Fb:2},a),shared:!1,upgrade:function(b,c,d){c(2)&&oo(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.h.indexNames.contains("newRequest")&&d.h.deleteIndex("newRequest"),vo(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.h.objectStoreNames.contains("sapisid")&&b.h.deleteObjectStore("sapisid");c(9)&&b.h.objectStoreNames.contains("SWHealthLog")&&b.h.deleteObjectStore("SWHealthLog")},
version:9});return Bq()}
;function Dq(a){return Io(Cq(),a)}
function Eq(a,b){var c,d,e,f;return A(function(g){if(g.h==1)return c={startTime:W(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},ticks:{}},g.yield(Dq(b),2);if(g.h!=3)return d=g.i,e=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:T("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),g.yield(qo(d,"LogsRequestsStore",e),3);f=g.i;c.ticks.tc=W();Fq(c);return g.return(f)})}
function Gq(a,b){var c,d,e,f,g,h,k,l;return A(function(m){if(m.h==1)return c={startTime:W(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},ticks:{}},m.yield(Dq(b),2);if(m.h!=3)return d=m.i,e=T("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=[a,e,0],g=[a,e,W()],h=IDBKeyRange.bound(f,g),k="prev",U("use_fifo_for_networkless")&&(k="next"),l=void 0,m.yield(no(d,["LogsRequestsStore"],{mode:"readwrite",la:!0},function(p){return Bo(p.objectStore("LogsRequestsStore").index("newRequestV2"),{query:h,direction:k},
function(t){t.getValue()&&(l=t.getValue(),a==="NEW"&&(l.status="QUEUED",t.update(l)))})}),3);
c.ticks.tc=W();Fq(c);return m.return(l)})}
function Hq(a,b){var c;return A(function(d){if(d.h==1)return d.yield(Dq(b),2);c=d.i;return d.return(no(c,["LogsRequestsStore"],{mode:"readwrite",la:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",jo(f.h.put(g,void 0)).then(function(){return g})})}))})}
function Iq(a,b,c,d){c=c===void 0?!0:c;var e;return A(function(f){if(f.h==1)return f.yield(Dq(b),2);e=f.i;return f.return(no(e,["LogsRequestsStore"],{mode:"readwrite",la:!0},function(g){var h=g.objectStore("LogsRequestsStore");return h.get(a).then(function(k){return k?(k.status="NEW",c&&(k.sendCount+=1),d!==void 0&&(k.options.compress=d),jo(h.h.put(k,void 0)).then(function(){return k})):co.resolve(void 0)})}))})}
function Jq(a,b){var c;return A(function(d){if(d.h==1)return d.yield(Dq(b),2);c=d.i;return d.return(c.delete("LogsRequestsStore",a))})}
function Kq(a){var b,c;return A(function(d){if(d.h==1)return d.yield(Dq(a),2);b=d.i;c=W()-2592E6;return d.yield(no(b,["LogsRequestsStore"],{mode:"readwrite",la:!0},function(e){return xo(e.objectStore("LogsRequestsStore"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return yo(f)})})}),0)})}
function Lq(){A(function(a){return a.yield(ep(),0)})}
function Fq(a){U("nwl_csi_killswitch")||Pp("networkless_performance",a,{sampleRate:1})}
;var Mq={accountStateChangeSignedIn:23,accountStateChangeSignedOut:24,delayedEventMetricCaptured:11,latencyActionBaselined:6,latencyActionInfo:7,latencyActionTicked:5,offlineTransferStatusChanged:2,offlineImageDownload:335,playbackStartStateChanged:9,systemHealthCaptured:3,mangoOnboardingCompleted:10,mangoPushNotificationReceived:230,mangoUnforkDbMigrationError:121,mangoUnforkDbMigrationSummary:122,mangoUnforkDbMigrationPreunforkDbVersionNumber:133,mangoUnforkDbMigrationPhoneMetadata:134,mangoUnforkDbMigrationPhoneStorage:135,
mangoUnforkDbMigrationStep:142,mangoAsyncApiMigrationEvent:223,mangoDownloadVideoResult:224,mangoHomepageVideoCount:279,mangoHomeV3State:295,mangoImageClientCacheHitEvent:273,sdCardStatusChanged:98,framesDropped:12,thumbnailHovered:13,deviceRetentionInfoCaptured:14,thumbnailLoaded:15,backToAppEvent:318,streamingStatsCaptured:17,offlineVideoShared:19,appCrashed:20,youThere:21,offlineStateSnapshot:22,mdxSessionStarted:25,mdxSessionConnected:26,mdxSessionDisconnected:27,bedrockResourceConsumptionSnapshot:28,
nextGenWatchWatchSwiped:29,kidsAccountsSnapshot:30,zeroStepChannelCreated:31,tvhtml5SearchCompleted:32,offlineSharePairing:34,offlineShareUnlock:35,mdxRouteDistributionSnapshot:36,bedrockRepetitiveActionTimed:37,unpluggedDegradationInfo:229,uploadMp4HeaderMoved:38,uploadVideoTranscoded:39,uploadProcessorStarted:46,uploadProcessorEnded:47,uploadProcessorReady:94,uploadProcessorRequirementPending:95,uploadProcessorInterrupted:96,uploadFrontendEvent:241,assetPackDownloadStarted:41,assetPackDownloaded:42,
assetPackApplied:43,assetPackDeleted:44,appInstallAttributionEvent:459,playbackSessionStopped:45,adBlockerMessagingShown:48,distributionChannelCaptured:49,dataPlanCpidRequested:51,detailedNetworkTypeCaptured:52,sendStateUpdated:53,receiveStateUpdated:54,sendDebugStateUpdated:55,receiveDebugStateUpdated:56,kidsErrored:57,mdxMsnSessionStatsFinished:58,appSettingsCaptured:59,mdxWebSocketServerHttpError:60,mdxWebSocketServer:61,startupCrashesDetected:62,coldStartInfo:435,offlinePlaybackStarted:63,liveChatMessageSent:225,
liveChatUserPresent:434,liveChatBeingModerated:457,liveCreationCameraUpdated:64,liveCreationEncodingCaptured:65,liveCreationError:66,liveCreationHealthUpdated:67,liveCreationVideoEffectsCaptured:68,liveCreationStageOccured:75,liveCreationBroadcastScheduled:123,liveCreationArchiveReplacement:149,liveCreationCostreamingConnection:421,liveCreationStreamWebrtcStats:288,mdxSessionRecoveryStarted:69,mdxSessionRecoveryCompleted:70,mdxSessionRecoveryStopped:71,visualElementShown:72,visualElementHidden:73,
visualElementGestured:78,visualElementStateChanged:208,screenCreated:156,playbackAssociated:202,visualElementAttached:215,playbackContextEvent:214,cloudCastingPlaybackStarted:74,webPlayerApiCalled:76,tvhtml5AccountDialogOpened:79,foregroundHeartbeat:80,foregroundHeartbeatScreenAssociated:111,kidsOfflineSnapshot:81,mdxEncryptionSessionStatsFinished:82,playerRequestCompleted:83,liteSchedulerStatistics:84,mdxSignIn:85,spacecastMetadataLookupRequested:86,spacecastBatchLookupRequested:87,spacecastSummaryRequested:88,
spacecastPlayback:89,spacecastDiscovery:90,tvhtml5LaunchUrlComponentChanged:91,mdxBackgroundPlaybackRequestCompleted:92,mdxBrokenAdditionalDataDeviceDetected:93,tvhtml5LocalStorage:97,tvhtml5DeviceStorageStatus:147,autoCaptionsAvailable:99,playbackScrubbingEvent:339,flexyState:100,interfaceOrientationCaptured:101,mainAppBrowseFragmentCache:102,offlineCacheVerificationFailure:103,offlinePlaybackExceptionDigest:217,vrCopresenceStats:104,vrCopresenceSyncStats:130,vrCopresenceCommsStats:137,vrCopresencePartyStats:153,
vrCopresenceEmojiStats:213,vrCopresenceEvent:141,vrCopresenceFlowTransitEvent:160,vrCowatchPartyEvent:492,vrPlaybackEvent:345,kidsAgeGateTracking:105,offlineDelayAllowedTracking:106,mainAppAutoOfflineState:107,videoAsThumbnailDownload:108,videoAsThumbnailPlayback:109,liteShowMore:110,renderingError:118,kidsProfilePinGateTracking:119,abrTrajectory:124,scrollEvent:125,streamzIncremented:126,kidsProfileSwitcherTracking:127,kidsProfileCreationTracking:129,buyFlowStarted:136,mbsConnectionInitiated:138,
mbsPlaybackInitiated:139,mbsLoadChildren:140,liteProfileFetcher:144,mdxRemoteTransaction:146,reelPlaybackError:148,reachabilityDetectionEvent:150,mobilePlaybackEvent:151,courtsidePlayerStateChanged:152,musicPersistentCacheChecked:154,musicPersistentCacheCleared:155,playbackInterrupted:157,playbackInterruptionResolved:158,fixFopFlow:159,anrDetection:161,backstagePostCreationFlowEnded:162,clientError:163,gamingAccountLinkStatusChanged:164,liteHousewarming:165,buyFlowEvent:167,kidsParentalGateTracking:168,
kidsSignedOutSettingsStatus:437,kidsSignedOutPauseHistoryFixStatus:438,tvhtml5WatchdogViolation:444,ypcUpgradeFlow:169,yongleStudy:170,ypcUpdateFlowStarted:171,ypcUpdateFlowCancelled:172,ypcUpdateFlowSucceeded:173,ypcUpdateFlowFailed:174,liteGrowthkitPromo:175,paymentFlowStarted:341,transactionFlowShowPaymentDialog:405,transactionFlowStarted:176,transactionFlowSecondaryDeviceStarted:222,transactionFlowSecondaryDeviceSignedOutStarted:383,transactionFlowCancelled:177,transactionFlowPaymentCallBackReceived:387,
transactionFlowPaymentSubmitted:460,transactionFlowPaymentSucceeded:329,transactionFlowSucceeded:178,transactionFlowFailed:179,transactionFlowPlayBillingConnectionStartEvent:428,transactionFlowSecondaryDeviceSuccess:458,transactionFlowErrorEvent:411,liteVideoQualityChanged:180,watchBreakEnablementSettingEvent:181,watchBreakFrequencySettingEvent:182,videoEffectsCameraPerformanceMetrics:183,adNotify:184,startupTelemetry:185,playbackOfflineFallbackUsed:186,outOfMemory:187,ypcPauseFlowStarted:188,ypcPauseFlowCancelled:189,
ypcPauseFlowSucceeded:190,ypcPauseFlowFailed:191,uploadFileSelected:192,ypcResumeFlowStarted:193,ypcResumeFlowCancelled:194,ypcResumeFlowSucceeded:195,ypcResumeFlowFailed:196,adsClientStateChange:197,ypcCancelFlowStarted:198,ypcCancelFlowCancelled:199,ypcCancelFlowSucceeded:200,ypcCancelFlowFailed:201,ypcCancelFlowGoToPaymentProcessor:402,ypcDeactivateFlowStarted:320,ypcRedeemFlowStarted:203,ypcRedeemFlowCancelled:204,ypcRedeemFlowSucceeded:205,ypcRedeemFlowFailed:206,ypcFamilyCreateFlowStarted:258,
ypcFamilyCreateFlowCancelled:259,ypcFamilyCreateFlowSucceeded:260,ypcFamilyCreateFlowFailed:261,ypcFamilyManageFlowStarted:262,ypcFamilyManageFlowCancelled:263,ypcFamilyManageFlowSucceeded:264,ypcFamilyManageFlowFailed:265,restoreContextEvent:207,embedsAdEvent:327,autoplayTriggered:209,clientDataErrorEvent:210,experimentalVssValidation:211,tvhtml5TriggeredEvent:212,tvhtml5FrameworksFieldTrialResult:216,tvhtml5FrameworksFieldTrialStart:220,musicOfflinePreferences:218,watchTimeSegment:219,appWidthLayoutError:221,
accountRegistryChange:226,userMentionAutoCompleteBoxEvent:227,downloadRecommendationEnablementSettingEvent:228,musicPlaybackContentModeChangeEvent:231,offlineDbOpenCompleted:232,kidsFlowEvent:233,kidsFlowCorpusSelectedEvent:234,videoEffectsEvent:235,unpluggedOpsEogAnalyticsEvent:236,playbackAudioRouteEvent:237,interactionLoggingDebugModeError:238,offlineYtbRefreshed:239,kidsFlowError:240,musicAutoplayOnLaunchAttempted:242,deviceContextActivityEvent:243,deviceContextEvent:244,templateResolutionException:245,
musicSideloadedPlaylistServiceCalled:246,embedsStorageAccessNotChecked:247,embedsHasStorageAccessResult:248,embedsItpPlayedOnReload:249,embedsRequestStorageAccessResult:250,embedsShouldRequestStorageAccessResult:251,embedsRequestStorageAccessState:256,embedsRequestStorageAccessFailedState:257,embedsItpWatchLaterResult:266,searchSuggestDecodingPayloadFailure:252,siriShortcutActivated:253,tvhtml5KeyboardPerformance:254,latencyActionSpan:255,elementsLog:267,ytbFileOpened:268,tfliteModelError:269,apiTest:270,
yongleUsbSetup:271,touStrikeInterstitialEvent:272,liteStreamToSave:274,appBundleClientEvent:275,ytbFileCreationFailed:276,adNotifyFailure:278,ytbTransferFailed:280,blockingRequestFailed:281,liteAccountSelector:282,liteAccountUiCallbacks:283,dummyPayload:284,browseResponseValidationEvent:285,entitiesError:286,musicIosBackgroundFetch:287,mdxNotificationEvent:289,layersValidationError:290,musicPwaInstalled:291,liteAccountCleanup:292,html5PlayerHealthEvent:293,watchRestoreAttempt:294,liteAccountSignIn:296,
notaireEvent:298,kidsVoiceSearchEvent:299,adNotifyFilled:300,delayedEventDropped:301,analyticsSearchEvent:302,systemDarkThemeOptOutEvent:303,flowEvent:304,networkConnectivityBaselineEvent:305,ytbFileImported:306,downloadStreamUrlExpired:307,directSignInEvent:308,lyricImpressionEvent:309,accessibilityStateEvent:310,tokenRefreshEvent:311,genericAttestationExecution:312,tvhtml5VideoSeek:313,unpluggedAutoPause:314,scrubbingEvent:315,bedtimeReminderEvent:317,tvhtml5UnexpectedRestart:319,tvhtml5StabilityTraceEvent:478,
tvhtml5OperationHealth:467,tvhtml5WatchKeyEvent:321,voiceLanguageChanged:322,tvhtml5LiveChatStatus:323,parentToolsCorpusSelectedEvent:324,offerAdsEnrollmentInitiated:325,networkQualityIntervalEvent:326,deviceStartupMetrics:328,heartbeatActionPlayerTransitioned:330,tvhtml5Lifecycle:331,heartbeatActionPlayerHalted:332,adaptiveInlineMutedSettingEvent:333,mainAppLibraryLoadingState:334,thirdPartyLogMonitoringEvent:336,appShellAssetLoadReport:337,tvhtml5AndroidAttestation:338,tvhtml5StartupSoundEvent:340,
iosBackgroundRefreshTask:342,iosBackgroundProcessingTask:343,sliEventBatch:344,postImpressionEvent:346,musicSideloadedPlaylistExport:347,idbUnexpectedlyClosed:348,voiceSearchEvent:349,mdxSessionCastEvent:350,idbQuotaExceeded:351,idbTransactionEnded:352,idbTransactionAborted:353,tvhtml5KeyboardLogging:354,idbIsSupportedCompleted:355,creatorStudioMobileEvent:356,idbDataCorrupted:357,parentToolsAppChosenEvent:358,webViewBottomSheetResized:359,activeStateControllerScrollPerformanceSummary:360,navigatorValidation:361,
mdxSessionHeartbeat:362,clientHintsPolyfillDiagnostics:363,clientHintsPolyfillEvent:364,proofOfOriginTokenError:365,kidsAddedAccountSummary:366,musicWearableDevice:367,ypcRefundFlowEvent:368,tvhtml5PlaybackMeasurementEvent:369,tvhtml5WatermarkMeasurementEvent:370,clientExpGcfPropagationEvent:371,mainAppReferrerIntent:372,leaderLockEnded:373,leaderLockAcquired:374,googleHatsEvent:375,persistentLensLaunchEvent:376,parentToolsChildWelcomeChosenEvent:378,browseThumbnailPreloadEvent:379,finalPayload:380,
mdxDialAdditionalDataUpdateEvent:381,webOrchestrationTaskLifecycleRecord:382,startupSignalEvent:384,accountError:385,gmsDeviceCheckEvent:386,accountSelectorEvent:388,accountUiCallbacks:389,mdxDialAdditionalDataProbeEvent:390,downloadsSearchIcingApiStats:391,downloadsSearchIndexUpdatedEvent:397,downloadsSearchIndexSnapshot:398,dataPushClientEvent:392,kidsCategorySelectedEvent:393,mdxDeviceManagementSnapshotEvent:394,prefetchRequested:395,prefetchableCommandExecuted:396,gelDebuggingEvent:399,webLinkTtsPlayEnd:400,
clipViewInvalid:401,persistentStorageStateChecked:403,cacheWipeoutEvent:404,playerEvent:410,sfvEffectPipelineStartedEvent:412,sfvEffectPipelinePausedEvent:429,sfvEffectPipelineEndedEvent:413,sfvEffectChosenEvent:414,sfvEffectLoadedEvent:415,sfvEffectUserInteractionEvent:465,sfvEffectFirstFrameProcessedLatencyEvent:416,sfvEffectAggregatedFramesProcessedLatencyEvent:417,sfvEffectAggregatedFramesDroppedEvent:418,sfvEffectPipelineErrorEvent:430,sfvEffectGraphFrozenEvent:419,sfvEffectGlThreadBlockedEvent:420,
mdeVideoChangedEvent:442,mdePlayerPerformanceMetrics:472,mdeExporterEvent:497,genericClientExperimentEvent:423,homePreloadTaskScheduled:424,homePreloadTaskExecuted:425,homePreloadCacheHit:426,polymerPropertyChangedInObserver:427,applicationStarted:431,networkCronetRttBatch:432,networkCronetRttSummary:433,repeatChapterLoopEvent:436,seekCancellationEvent:462,lockModeTimeoutEvent:483,externalVideoShareToYoutubeAttempt:501,offlineTransferStarted:4,musicOfflineMixtapePreferencesChanged:16,mangoDailyNewVideosNotificationAttempt:40,
mangoDailyNewVideosNotificationError:77,dtwsPlaybackStarted:112,dtwsTileFetchStarted:113,dtwsTileFetchCompleted:114,dtwsTileFetchStatusChanged:145,dtwsKeyframeDecoderBufferSent:115,dtwsTileUnderflowedOnNonkeyframe:116,dtwsBackfillFetchStatusChanged:143,dtwsBackfillUnderflowed:117,dtwsAdaptiveLevelChanged:128,blockingVisitorIdTimeout:277,liteSocial:18,mobileJsInvocation:297,biscottiBasedDetection:439,coWatchStateChange:440,embedsVideoDataDidChange:441,shortsFirst:443,cruiseControlEvent:445,qoeClientLoggingContext:446,
atvRecommendationJobExecuted:447,tvhtml5UserFeedback:448,producerProjectCreated:449,producerProjectOpened:450,producerProjectDeleted:451,producerProjectElementAdded:453,producerProjectElementRemoved:454,tvhtml5ShowClockEvent:455,deviceCapabilityCheckMetrics:456,youtubeClearcutEvent:461,offlineBrowseFallbackEvent:463,getCtvTokenEvent:464,startupDroppedFramesSummary:466,screenshotEvent:468,miniAppPlayEvent:469,elementsDebugCounters:470,fontLoadEvent:471,webKillswitchReceived:473,webKillswitchExecuted:474,
cameraOpenEvent:475,manualSmoothnessMeasurement:476,tvhtml5AppQualityEvent:477,polymerPropertyAccessEvent:479,miniAppSdkUsage:480,cobaltTelemetryEvent:481,crossDevicePlayback:482,channelCreatedWithObakeImage:484,channelEditedWithObakeImage:485,offlineDeleteEvent:486,crossDeviceNotificationTransfer:487,androidIntentEvent:488,unpluggedAmbientInterludesCounterfactualEvent:489,keyPlaysPlayback:490,shortsCreationFallbackEvent:493,vssData:491,castMatch:494,miniAppPerformanceMetrics:495,userFeedbackEvent:496,
kidsGuestSessionMismatch:498,musicSideloadedPlaylistMigrationEvent:499,sleepTimerSessionFinishEvent:500};var Nq={},Oq=ip("ServiceWorkerLogsDatabase",{xb:(Nq.SWHealthLog={Fb:1},Nq),shared:!0,upgrade:function(a,b){b(1)&&vo(oo(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function Pq(a){return Io(Oq(),a)}
function Qq(a){var b,c;A(function(d){if(d.h==1)return d.yield(Pq(a),2);b=d.i;c=W()-2592E6;return d.yield(no(b,["SWHealthLog"],{mode:"readwrite",la:!0},function(e){return xo(e.objectStore("SWHealthLog"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return yo(f)})})}),0)})}
function Rq(a){var b;return A(function(c){if(c.h==1)return c.yield(Pq(a),2);b=c.i;return c.yield(b.clear("SWHealthLog"),0)})}
;var Sq={},Tq=0;function Uq(a){var b=new Image,c=""+Tq++;Sq[c]=b;b.onload=b.onerror=function(){delete Sq[c]};
b.src=a}
;var Vq;function Wq(){Vq||(Vq=new yn("yt.offline"));return Vq}
function Xq(a){if(U("offline_error_handling")){var b=Wq().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);Wq().set("errors",b,2592E3,!0)}}
;function Yq(){this.h=new Map;this.i=!1}
function Zq(){if(!Yq.h){var a=E("yt.networkRequestMonitor.instance")||new Yq;D("yt.networkRequestMonitor.instance",a);Yq.h=a}return Yq.h}
Yq.prototype.requestComplete=function(a,b){b&&(this.i=!0);a=this.removeParams(a);this.h.get(a)||this.h.set(a,b)};
Yq.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.h.get(a))?!1:a===!1&&this.i?!0:null};
Yq.prototype.removeParams=function(a){return a.split("?")[0]};
Yq.prototype.removeParams=Yq.prototype.removeParams;Yq.prototype.isEndpointCFR=Yq.prototype.isEndpointCFR;Yq.prototype.requestComplete=Yq.prototype.requestComplete;Yq.getInstance=Zq;function $q(){wd.call(this);var a=this;this.j=!1;this.i=bj();this.i.listen("networkstatus-online",function(){if(a.j&&U("offline_error_handling")){var b=Wq().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new V(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;ul(d)}Wq().set("errors",{},2592E3,!0)}}})}
y($q,wd);function ar(){if(!$q.h){var a=E("yt.networkStatusManager.instance")||new $q;D("yt.networkStatusManager.instance",a);$q.h=a}return $q.h}
n=$q.prototype;n.va=function(){return this.i.va()};
n.gb=function(a){this.i.i=a};
n.de=function(){var a=window.navigator.onLine;return a===void 0?!0:a};
n.Ud=function(){this.j=!0};
n.listen=function(a,b){return this.i.listen(a,b)};
n.pc=function(a){a=$i(this.i,a);a.then(function(b){U("use_cfr_monitor")&&Zq().requestComplete("generate_204",b)});
return a};
$q.prototype.sendNetworkCheckRequest=$q.prototype.pc;$q.prototype.listen=$q.prototype.listen;$q.prototype.enableErrorFlushing=$q.prototype.Ud;$q.prototype.getWindowStatus=$q.prototype.de;$q.prototype.networkStatusHint=$q.prototype.gb;$q.prototype.isNetworkAvailable=$q.prototype.va;$q.getInstance=ar;function br(a){a=a===void 0?{}:a;wd.call(this);var b=this;this.i=this.m=0;this.j=ar();var c=E("yt.networkStatusManager.instance.listen").bind(this.j);c&&(a.rateLimit?(this.rateLimit=a.rateLimit,c("networkstatus-online",function(){cr(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){cr(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){xd(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){xd(b,"publicytnetworkstatus-offline")})))}
y(br,wd);br.prototype.va=function(){var a=E("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.j)():!0};
br.prototype.gb=function(a){var b=E("yt.networkStatusManager.instance.networkStatusHint").bind(this.j);b&&b(a)};
br.prototype.pc=function(a){var b=this,c;return A(function(d){c=E("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.j);return U("skip_network_check_if_cfr")&&Zq().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.gb(((f=window.navigator)==null?void 0:f.onLine)||!0);e(b.va())})):c?d.return(c(a)):d.return(!0)})};
function cr(a,b){a.rateLimit?a.i?(cj.qa(a.m),a.m=cj.pa(function(){a.l!==b&&(xd(a,b),a.l=b,a.i=W())},a.rateLimit-(W()-a.i))):(xd(a,b),a.l=b,a.i=W()):xd(a,b)}
;var dr;function er(){var a=rq.call;dr||(dr=new br({Pg:!0,Ig:!0}));a.call(rq,this,{ha:{Pd:Kq,ob:Jq,dd:Gq,se:Hq,Lc:Iq,set:Eq},fa:dr,handleError:function(b,c,d){var e,f=d==null?void 0:(e=d.error)==null?void 0:e.code;if(f===400||f===415){var g;vl(new V(b.message,c,d==null?void 0:(g=d.error)==null?void 0:g.code),void 0,void 0,void 0,!0)}else ul(b)},
pb:vl,sendFn:fr,now:W,Gd:Xq,Da:xn(),Kc:"publicytnetworkstatus-online",Hc:"publicytnetworkstatus-offline",dc:!0,ac:.1,mc:Ol("potential_esf_error_limit",10),Y:U,Ib:!(Pm()&&gr())});this.j=new Mi;U("networkless_immediately_drop_all_requests")&&Lq();fp("LogsDatabaseV2")}
y(er,rq);function hr(){var a=E("yt.networklessRequestController.instance");a||(a=new er,D("yt.networklessRequestController.instance",a),U("networkless_logging")&&Vo().then(function(b){a.X=b;tq(a);a.j.resolve();a.dc&&Math.random()<=a.ac&&a.X&&Qq(a.X);U("networkless_immediately_drop_sw_health_store")&&ir(a)}));
return a}
er.prototype.writeThenSend=function(a,b){b||(b={});b=jr(a,b);Pm()||(this.h=!1);rq.prototype.writeThenSend.call(this,a,b)};
er.prototype.sendThenWrite=function(a,b,c){b||(b={});b=jr(a,b);Pm()||(this.h=!1);rq.prototype.sendThenWrite.call(this,a,b,c)};
er.prototype.sendAndWrite=function(a,b){b||(b={});b=jr(a,b);Pm()||(this.h=!1);rq.prototype.sendAndWrite.call(this,a,b)};
er.prototype.awaitInitialization=function(){return this.j.promise};
function ir(a){var b;A(function(c){if(!a.X)throw b=Zn("clearSWHealthLogsDb"),b;return c.return(Rq(a.X).catch(function(d){a.handleError(d)}))})}
function fr(a,b,c,d){d=d===void 0?!1:d;b=U("web_fp_via_jspb")?Object.assign({},b):b;U("use_cfr_monitor")&&kr(a,b);if(U("use_request_time_ms_header"))b.headers&&Fl(a)&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(W())));else{var e;if((e=b.postParams)==null?0:e.requestTimeMs)b.postParams.requestTimeMs=Math.round(W())}if(c&&Object.keys(b).length===0){var f=f===void 0?"":f;var g=g===void 0?!1:g;var h=h===void 0?!1:h;if(a)if(f)Tl(a,void 0,"POST",f,void 0);else if(T("USE_NET_AJAX_FOR_PING_TRANSPORT",
!1)||h)Tl(a,void 0,"GET","",void 0,void 0,g,h);else{b:{try{var k=new cb({url:a});if(k.j&&k.i||k.l){var l=mc(nc(5,a)),m;if(!(m=!l||!l.endsWith("/aclk"))){var p=a.search(vc),t=uc(a,0,"ri",p);if(t<0)var r=null;else{var v=a.indexOf("&",t);if(v<0||v>p)v=p;r=decodeURIComponent(a.slice(t+3,v!==-1?v:0).replace(/\+/g," "))}m=r!=="1"}var u=!m;break b}}catch(F){}u=!1}if(u){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var z=!0;break b}}catch(F){}z=!1}c=z?!0:!1}else c=
!1;c||Uq(a)}}else b.compress?b.postBody?(typeof b.postBody!=="string"&&(b.postBody=JSON.stringify(b.postBody)),gq(a,b.postBody,b,Xl,d)):gq(a,JSON.stringify(b.postParams),b,Wl,d):Xl(a,b)}
function jr(a,b){U("use_event_time_ms_header")&&Fl(a)&&(b.headers||(b.headers={}),b.headers["X-Goog-Event-Time"]=JSON.stringify(Math.round(W())));return b}
function kr(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){Zq().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){Zq().requestComplete(a,!0);d(e,f)}}
function gr(){return oc(document.location.toString())!=="www.youtube-nocookie.com"}
;var lr=!1,mr=C.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:lr};D("ytNetworklessLoggingInitializationOptions",mr);function nr(){var a;A(function(b){if(b.h==1)return b.yield(Vo(),2);a=b.i;if(!a||!Pm()&&!U("nwl_init_require_datasync_id_killswitch")||!gr())return b.B(0);lr=!0;mr.isNwlInitialized=lr;return b.yield(hr().awaitInitialization(),0)})}
;function or(a){var b=this;this.config_=null;a?this.config_=a:wp()&&(this.config_=xp());Sm(function(){pq(b)},5E3)}
or.prototype.isReady=function(){!this.config_&&wp()&&(this.config_=xp());return!!this.config_};
function qq(a,b,c,d){function e(v){v=v===void 0?!1:v;var u;if(d.retry&&h!="www.youtube-nocookie.com"&&(v||U("skip_ls_gel_retry")||g.headers["Content-Type"]!=="application/json"||(u=nq(b,c,l,k)),u)){var z=g.onSuccess,F=g.onFetchSuccess;g.onSuccess=function(S,da){oq(u);z(S,da)};
c.onFetchSuccess=function(S,da){oq(u);F(S,da)}}try{if(v&&d.retry&&!d.networklessOptions.bypassNetworkless)g.method="POST",d.networklessOptions.writeThenSend?hr().writeThenSend(r,g):hr().sendAndWrite(r,g);
else if(d.compress){var J=!d.networklessOptions.writeThenSend;if(g.postBody){var O=g.postBody;typeof O!=="string"&&(O=JSON.stringify(g.postBody));gq(r,O,g,Xl,J)}else gq(r,JSON.stringify(g.postParams),g,Wl,J)}else U("web_all_payloads_via_jspb")?Xl(r,g):Wl(r,g)}catch(S){if(S.name==="InvalidAccessError")u&&(oq(u),u=0),vl(Error("An extension is blocking network request."));else throw S;}u&&Sm(function(){pq(a)},5E3)}
!T("VISITOR_DATA")&&b!=="visitor_id"&&Math.random()<.01&&vl(new V("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new V("innertube xhrclient not ready",b,c,d);ul(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(v,u){if(d.onSuccess)d.onSuccess(u)},
onFetchSuccess:function(v){if(d.onSuccess)d.onSuccess(v)},
onError:function(v,u){if(d.onError)d.onError(u)},
onFetchError:function(v){if(d.onError)d.onError(v)},
timeout:d.timeout,withCredentials:!0,compress:d.compress};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.me)&&(h=f);var k=a.config_.oe||!1,l=zp(k,h,d);Object.assign(g.headers,l);(f=g.headers.Authorization)&&!h&&k&&(g.headers["x-origin"]=window.location.origin);var m="/youtubei/"+a.config_.innertubeApiVersion+"/"+b,p={alt:"json"},t=a.config_.ne&&f;t=t&&f.startsWith("Bearer");t||(p.key=a.config_.innertubeApiKey);var r=El(""+h+m,p||{},!0);(E("ytNetworklessLoggingInitializationOptions")?
mr.isNwlInitialized:lr)?To().then(function(v){e(v)}):e(!1)}
;var pr=0,qr=Yc?"webkit":Xc?"moz":Vc?"ms":Uc?"o":"";D("ytDomDomGetNextId",E("ytDomDomGetNextId")||function(){return++pr});var rr={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function sr(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.scale=1;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in rr||(this[b]=a[b]);this.scale=a.scale;this.rotation=a.rotation;var c=a.target||a.srcElement;c&&c.nodeType==3&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;
if(d)try{d=d.nodeName?d:null}catch(e){d=null}else this.type=="mouseover"?d=a.fromElement:this.type=="mouseout"&&(d=a.toElement);this.relatedTarget=d;this.clientX=a.clientX!=void 0?a.clientX:a.pageX;this.clientY=a.clientY!=void 0?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||(this.type=="keypress"?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.h=a.pageX;this.i=a.pageY}}catch(e){}}
function tr(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.h=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.i=a.clientY+b}}
sr.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
sr.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
sr.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var Mb=C.ytEventsEventsListeners||{};D("ytEventsEventsListeners",Mb);var ur=C.ytEventsEventsCounter||{count:0};D("ytEventsEventsCounter",ur);
function vr(a,b,c,d){d=d===void 0?{}:d;a.addEventListener&&(b!="mouseenter"||"onmouseenter"in document?b!="mouseleave"||"onmouseenter"in document?b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return Lb(function(e){var f=typeof e[4]==="boolean"&&e[4]==!!d,g=Ra(e[4])&&Ra(d)&&Qb(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
function wr(a,b,c,d){d=d===void 0?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=vr(a,b,c,d);if(e)return e;e=++ur.count+"";var f=!(b!="mouseenter"&&b!="mouseleave"||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new sr(h);if(!Hd(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new sr(h);
h.currentTarget=a;return c.call(a,h)};
g=tl(g);a.addEventListener?(b=="mouseenter"&&f?b="mouseover":b=="mouseleave"&&f?b="mouseout":b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),xr()||typeof d==="boolean"?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);Mb[e]=[a,b,c,g,d];return e}
function yr(a){a&&(typeof a=="string"&&(a=[a]),Db(a,function(b){if(b in Mb){var c=Mb[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?xr()||typeof c==="boolean"?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete Mb[b]}}))}
var xr=Cd(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});function zr(a){this.H=a;this.h=null;this.l=0;this.A=null;this.m=0;this.i=[];for(a=0;a<4;a++)this.i.push(0);this.j=0;this.W=wr(window,"mousemove",Xa(this.ba,this));a=Xa(this.P,this);typeof a==="function"&&(a=tl(a));this.da=window.setInterval(a,25)}
$a(zr,G);zr.prototype.ba=function(a){a.h===void 0&&tr(a);var b=a.h;a.i===void 0&&tr(a);this.h=new Dd(b,a.i)};
zr.prototype.P=function(){if(this.h){var a=W();if(this.l!=0){var b=this.A,c=this.h,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.l);this.i[this.j]=Math.abs((d-this.m)/this.m)>.5?1:0;for(c=b=0;c<4;c++)b+=this.i[c]||0;b>=3&&this.H();this.m=d}this.l=a;this.A=this.h;this.j=(this.j+1)%4}};
zr.prototype.U=function(){window.clearInterval(this.da);yr(this.W)};var Dr={};
function Er(a){var b=a===void 0?{}:a;a=b.Ce===void 0?!1:b.Ce;b=b.Vd===void 0?!0:b.Vd;if(E("_lact",window)==null){var c=parseInt(T("LACT"),10);c=isFinite(c)?Date.now()-Math.max(c,0):-1;D("_lact",c,window);D("_fact",c,window);c==-1&&Fr();wr(document,"keydown",Fr);wr(document,"keyup",Fr);wr(document,"mousedown",Fr);wr(document,"mouseup",Fr);a?wr(window,"touchmove",function(){Gr("touchmove",200)},{passive:!0}):(wr(window,"resize",function(){Gr("resize",200)}),b&&wr(window,"scroll",function(){Gr("scroll",200)}));
new zr(function(){Gr("mouse",100)});
wr(document,"touchstart",Fr,{passive:!0});wr(document,"touchend",Fr,{passive:!0})}}
function Gr(a,b){Dr[a]||(Dr[a]=!0,cj.pa(function(){Fr();Dr[a]=!1},b))}
function Fr(){E("_lact",window)==null&&Er();var a=Date.now();D("_lact",a,window);E("_fact",window)==-1&&D("_fact",a,window);(a=E("ytglobal.ytUtilActivityCallback_"))&&a()}
function Hr(){var a=E("_lact",window);return a==null?-1:Math.max(Date.now()-a,0)}
;var Ir=C.ytPubsubPubsubInstance||new M,Jr=C.ytPubsubPubsubSubscribedKeys||{},Kr=C.ytPubsubPubsubTopicToKeys||{},Lr=C.ytPubsubPubsubIsSynchronous||{};function Mr(a,b){var c=Nr();if(c&&b){var d=c.subscribe(a,function(){function e(){Jr[d]&&b.apply&&typeof b.apply=="function"&&b.apply(window,f)}
var f=arguments;try{Lr[a]?e():Ll(e,0)}catch(g){ul(g)}},void 0);
Jr[d]=!0;Kr[a]||(Kr[a]=[]);Kr[a].push(d);return d}return 0}
function Or(a){var b=Nr();b&&(typeof a==="number"?a=[a]:typeof a==="string"&&(a=[parseInt(a,10)]),Db(a,function(c){b.unsubscribeByKey(c);delete Jr[c]}))}
function Pr(a,b){var c=Nr();c&&c.publish.apply(c,arguments)}
function Qr(a){var b=Nr();if(b)if(b.clear(a),a)Rr(a);else for(var c in Kr)Rr(c)}
function Nr(){return C.ytPubsubPubsubInstance}
function Rr(a){Kr[a]&&(a=Kr[a],Db(a,function(b){Jr[b]&&delete Jr[b]}),a.length=0)}
M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.Bb;M.prototype.publish=M.prototype.Ya;M.prototype.clear=M.prototype.clear;D("ytPubsubPubsubInstance",Ir);D("ytPubsubPubsubTopicToKeys",Kr);D("ytPubsubPubsubIsSynchronous",Lr);D("ytPubsubPubsubSubscribedKeys",Jr);var Sr=Symbol("injectionDeps");function Tr(a){this.name=a}
Tr.prototype.toString=function(){return"InjectionToken("+this.name+")"};
function Ur(a){this.key=a}
function Vr(){this.i=new Map;this.j=new Map;this.h=new Map}
function Wr(a,b){a.i.set(b.oc,b);var c=a.j.get(b.oc);if(c)try{c.Wg(a.resolve(b.oc))}catch(d){c.Ug(d)}}
Vr.prototype.resolve=function(a){return a instanceof Ur?Xr(this,a.key,[],!0):Xr(this,a,[])};
function Xr(a,b,c,d){d=d===void 0?!1:d;if(c.indexOf(b)>-1)throw Error("Deps cycle for: "+b);if(a.h.has(b))return a.h.get(b);if(!a.i.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.i.get(b);c.push(b);if(d.Cd!==void 0)var e=d.Cd;else if(d.mf)e=d[Sr]?Yr(a,d[Sr],c):[],e=d.mf.apply(d,la(e));else if(d.Bd){e=d.Bd;var f=e[Sr]?Yr(a,e[Sr],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(la(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.Zg||a.h.set(b,e);return e}
function Yr(a,b,c){return b?b.map(function(d){return d instanceof Ur?Xr(a,d.key,c,!0):Xr(a,d,c)}):[]}
;var Zr;function $r(){Zr||(Zr=new Vr);return Zr}
;var as=window;function bs(){var a,b;return"h5vcc"in as&&((a=as.h5vcc.traceEvent)==null?0:a.traceBegin)&&((b=as.h5vcc.traceEvent)==null?0:b.traceEnd)?1:"performance"in as&&as.performance.mark&&as.performance.measure?2:0}
function cs(a){var b=bs();switch(b){case 1:as.h5vcc.traceEvent.traceBegin("YTLR",a);break;case 2:as.performance.mark(a+"-start");break;case 0:break;default:Xb(b,"unknown trace type")}}
function ds(a){var b=bs();switch(b){case 1:as.h5vcc.traceEvent.traceEnd("YTLR",a);break;case 2:b=a+"-start";var c=a+"-end";as.performance.mark(c);as.performance.measure(a,b,c);break;case 0:break;default:Xb(b,"unknown trace type")}}
;var es=U("web_enable_lifecycle_monitoring")&&bs()!==0,gs=U("web_enable_lifecycle_monitoring");function hs(a){var b=this;var c=c===void 0?0:c;var d=d===void 0?xn():d;this.j=c;this.scheduler=d;this.i=new Mi;this.h=a;for(a={cb:0};a.cb<this.h.length;a={lc:void 0,cb:a.cb},a.cb++)a.lc=this.h[a.cb],c=function(e){return function(){e.lc.Ac();b.h[e.cb].nc=!0;b.h.every(function(f){return f.nc===!0})&&b.i.resolve()}}(a),d=this.getPriority(a.lc),d=this.scheduler.ab(c,d),this.h[a.cb]=Object.assign({},a.lc,{Ac:c,
jobId:d})}
function is(a){var b=Array.from(a.h.keys()).sort(function(d,e){return a.getPriority(a.h[e])-a.getPriority(a.h[d])});
b=x(b);for(var c=b.next();!c.done;c=b.next())c=a.h[c.value],c.jobId===void 0||c.nc||(a.scheduler.qa(c.jobId),a.scheduler.ab(c.Ac,10))}
hs.prototype.cancel=function(){for(var a=x(this.h),b=a.next();!b.done;b=a.next())b=b.value,b.jobId===void 0||b.nc||this.scheduler.qa(b.jobId),b.nc=!0;this.i.resolve()};
hs.prototype.getPriority=function(a){var b;return(b=a.priority)!=null?b:this.j};function js(a){this.state=a;this.plugins=[];this.l=void 0;this.A={};es&&cs(this.state)}
n=js.prototype;n.install=function(a){this.plugins.push(a);return this};
n.uninstall=function(){var a=this;B.apply(0,arguments).forEach(function(b){b=a.plugins.indexOf(b);b>-1&&a.plugins.splice(b,1)})};
n.transition=function(a,b){var c=this;es&&ds(this.state);var d=this.transitions.find(function(f){return Array.isArray(f.from)?f.from.find(function(g){return g===c.state&&f.to===a}):f.from===c.state&&f.to===a});
if(d){this.j&&(is(this.j),this.j=void 0);ks(this,a,b);this.state=a;es&&cs(this.state);d=d.action.bind(this);var e=this.plugins.filter(function(f){return f[a]}).map(function(f){return f[a]});
d(ls(this,e),b)}else throw Error("no transition specified from "+this.state+" to "+a);};
function ls(a,b){var c=b.filter(function(e){return ms(a,e)===10}),d=b.filter(function(e){return ms(a,e)!==10});
return a.A.Yg?function(){var e=B.apply(0,arguments);return A(function(f){if(f.h==1)return f.yield(a.Je.apply(a,[c].concat(la(e))),2);a.xd.apply(a,[d].concat(la(e)));f.h=0})}:function(){var e=B.apply(0,arguments);
a.Ke.apply(a,[c].concat(la(e)));a.xd.apply(a,[d].concat(la(e)))}}
n.Ke=function(a){for(var b=B.apply(1,arguments),c=xn(),d=x(a),e=d.next(),f={};!e.done;f={Lb:void 0},e=d.next())f.Lb=e.value,c.Db(function(g){return function(){ns(g.Lb.name);g.Lb.callback.apply(g.Lb,la(b));ps(g.Lb.name)}}(f))};
n.Je=function(a){var b=B.apply(1,arguments),c,d,e,f,g;return A(function(h){h.h==1&&(c=xn(),d=x(a),e=d.next(),f={});if(h.h!=3){if(e.done)return h.B(0);f.tb=e.value;f.Vb=void 0;g=function(k){return function(){ns(k.tb.name);var l=k.tb.callback.apply(k.tb,la(b));typeof(l==null?void 0:l.then)==="function"?k.Vb=l.then(function(){ps(k.tb.name)}):ps(k.tb.name)}}(f);
c.Db(g);return f.Vb?h.yield(f.Vb,3):h.B(3)}f={tb:void 0,Vb:void 0};e=d.next();return h.B(2)})};
n.xd=function(a){var b=B.apply(1,arguments),c=this,d=a.map(function(e){return{Ac:function(){ns(e.name);e.callback.apply(e,la(b));ps(e.name)},
priority:ms(c,e)}});
d.length&&(this.j=new hs(d))};
function ms(a,b){var c,d;return(d=(c=a.l)!=null?c:b.priority)!=null?d:0}
function ns(a){es&&a&&cs(a)}
function ps(a){es&&a&&ds(a)}
function ks(a,b,c){gs&&console.groupCollapsed&&console.groupEnd&&(console.groupCollapsed("["+a.constructor.name+"] '"+a.state+"' to '"+b+"'"),console.log("with message: ",c),console.groupEnd())}
fa.Object.defineProperties(js.prototype,{currentState:{configurable:!0,enumerable:!0,get:function(){return this.state}}});function qs(a){js.call(this,a===void 0?"none":a);this.h=null;this.l=10;this.transitions=[{from:"none",to:"application_navigating",action:this.i},{from:"application_navigating",to:"none",action:this.v},{from:"application_navigating",to:"application_navigating",action:function(){}},
{from:"none",to:"none",action:function(){}}]}
var rs;y(qs,js);qs.prototype.i=function(a,b){var c=this;this.h=Sm(function(){c.currentState==="application_navigating"&&c.transition("none")},5E3);
a(b==null?void 0:b.event)};
qs.prototype.v=function(a,b){this.h&&(cj.qa(this.h),this.h=null);a(b==null?void 0:b.event)};
function ss(){rs||(rs=new qs);return rs}
;var ts=[];D("yt.logging.transport.getScrapedGelPayloads",function(){return ts});function us(){this.store={};this.h={}}
us.prototype.storePayload=function(a,b){a=vs(a);this.store[a]?this.store[a].push(b):(this.h={},this.store[a]=[b]);return a};
us.prototype.smartExtractMatchingEntries=function(a){if(!a.keys.length)return[];for(var b=ws(this,a.keys.splice(0,1)[0]),c=[],d=0;d<b.length;d++)this.store[b[d]]&&a.sizeLimit&&(this.store[b[d]].length<=a.sizeLimit?(c.push.apply(c,la(this.store[b[d]])),delete this.store[b[d]]):c.push.apply(c,la(this.store[b[d]].splice(0,a.sizeLimit))));(a==null?0:a.sizeLimit)&&c.length<(a==null?void 0:a.sizeLimit)&&(a.sizeLimit-=c.length,c.push.apply(c,la(this.smartExtractMatchingEntries(a))));return c};
us.prototype.extractMatchingEntries=function(a){a=ws(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,la(this.store[a[c]])),delete this.store[a[c]]);return b};
us.prototype.getSequenceCount=function(a){a=ws(this,a);for(var b=0,c=0;c<a.length;c++){var d=void 0;b+=((d=this.store[a[c]])==null?void 0:d.length)||0}return b};
function ws(a,b){var c=vs(b);if(a.h[c])return a.h[c];var d=Object.keys(a.store)||[];if(d.length<=1&&vs(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(xs(b.auth,g[0])){var h=b.isJspb;xs(h===void 0?"undefined":h?"true":"false",g[1])&&xs(b.cttAuthInfo,g[2])&&(h=b.tier,h=h===void 0?"undefined":JSON.stringify(h),xs(h,g[3])&&e.push(d[f]))}}return a.h[c]=e}
function xs(a,b){return a===void 0||a==="undefined"?!0:a===b}
us.prototype.getSequenceCount=us.prototype.getSequenceCount;us.prototype.extractMatchingEntries=us.prototype.extractMatchingEntries;us.prototype.smartExtractMatchingEntries=us.prototype.smartExtractMatchingEntries;us.prototype.storePayload=us.prototype.storePayload;function vs(a){return[a.auth===void 0?"undefined":a.auth,a.isJspb===void 0?"undefined":a.isJspb,a.cttAuthInfo===void 0?"undefined":a.cttAuthInfo,a.tier===void 0?"undefined":a.tier].join("/")}
;function ys(a,b){if(a)return a[b.name]}
;var zs=Ol("initial_gel_batch_timeout",2E3),As=Ol("gel_queue_timeout_max_ms",6E4),Bs=Math.pow(2,16)-1,Cs=Ol("gel_min_batch_size",5),Ds=void 0;function Es(){this.l=this.h=this.i=0;this.j=!1}
var Fs=new Es,Gs=new Es,Hs=new Es,Is=new Es,Js,Ks=!0,Ls=C.ytLoggingTransportTokensToCttTargetIds_||{};D("ytLoggingTransportTokensToCttTargetIds_",Ls);var Ms={};function Ns(){var a=E("yt.logging.ims");a||(a=new us,D("yt.logging.ims",a));return a}
function Os(a,b){if(a.endpoint==="log_event"){Ps();var c=Qs(a),d=Rs(a.payload)||"";a:{if(U("enable_web_tiered_gel")){var e=Mq[d||""];var f,g,h,k=$r().resolve(new Ur(rp))==null?void 0:(f=sp())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.eventLoggingConfig)==null?void 0:h.payloadPolicies;if(k)for(f=0;f<k.length;f++)if(k[f].payloadNumber===e){e=k[f];break a}}e=void 0}k=200;if(e){if(e.enabled===!1&&!U("web_payload_policy_disabled_killswitch"))return;k=Ss(e.tier);if(k===400){Ts(a,b);return}}Ms[c]=
!0;e={cttAuthInfo:c,isJspb:!1,tier:k};Ns().storePayload(e,a.payload);Us(b,c,e,d==="gelDebuggingEvent")}}
function Us(a,b,c,d){function e(){Vs({writeThenSend:!0},U("flush_only_full_queue")?b:void 0,f,c.tier)}
var f=!1;f=f===void 0?!1:f;d=d===void 0?!1:d;a&&(Ds=new a);a=Ol("tvhtml5_logging_max_batch_ads_fork")||Ol("web_logging_max_batch")||100;var g=W(),h=Ws(f,c.tier),k=h.l;d&&(h.j=!0);d=0;c&&(d=Ns().getSequenceCount(c));d>=1E3?e():d>=a?Js||(Js=Xs(function(){e();Js=void 0},0)):g-k>=10&&(Ys(f,c.tier),h.l=g)}
function Ts(a,b){if(a.endpoint==="log_event"){Ps();var c=Qs(a),d=new Map;d.set(c,[a.payload]);var e=Rs(a.payload)||"";b&&(Ds=new b);return new Ud(function(f,g){Ds&&Ds.isReady()?Zs(d,Ds,f,g,{bypassNetworkless:!0},!0,e==="gelDebuggingEvent"):f()})}}
function Qs(a){var b="";if(a.dangerousLogToVisitorSession)b="visitorOnlyApprovedKey";else if(a.cttAuthInfo){b=a.cttAuthInfo;var c={};b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId);Ls[a.cttAuthInfo.token]=c;b=a.cttAuthInfo.token}return b}
function Vs(a,b,c,d){a=a===void 0?{}:a;c=c===void 0?!1:c;new Ud(function(e,f){var g=Ws(c,d),h=g.j;g.j=!1;$s(g.i);$s(g.h);g.h=0;Ds&&Ds.isReady()?d===void 0&&U("enable_web_tiered_gel")?at(e,f,a,b,c,300,h):at(e,f,a,b,c,d,h):(Ys(c,d),e())})}
function at(a,b,c,d,e,f,g){var h=Ds;c=c===void 0?{}:c;e=e===void 0?!1:e;f=f===void 0?200:f;g=g===void 0?!1:g;var k=new Map,l={isJspb:e,cttAuthInfo:d,tier:f};e={isJspb:e,cttAuthInfo:d};if(d!==void 0)f=U("enable_web_tiered_gel")?Ns().smartExtractMatchingEntries({keys:[l,e],sizeLimit:1E3}):Ns().extractMatchingEntries(e),k.set(d,f);else for(d=x(Object.keys(Ms)),l=d.next();!l.done;l=d.next())l=l.value,e=U("enable_web_tiered_gel")?Ns().smartExtractMatchingEntries({keys:[{isJspb:!1,cttAuthInfo:l,tier:f},
{isJspb:!1,cttAuthInfo:l}],sizeLimit:1E3}):Ns().extractMatchingEntries({isJspb:!1,cttAuthInfo:l}),e.length>0&&k.set(l,e),(U("web_fp_via_jspb_and_json")&&c.writeThenSend||!U("web_fp_via_jspb_and_json"))&&delete Ms[l];Zs(k,h,a,b,c,!1,g)}
function Ys(a,b){function c(){Vs({writeThenSend:!0},void 0,a,b)}
a=a===void 0?!1:a;b=b===void 0?200:b;var d=Ws(a,b),e=d===Is||d===Hs?5E3:As;U("web_gel_timeout_cap")&&!d.h&&(e=Xs(function(){c()},e),d.h=e);
$s(d.i);e=T("LOGGING_BATCH_TIMEOUT",Ol("web_gel_debounce_ms",1E4));U("shorten_initial_gel_batch_timeout")&&Ks&&(e=zs);e=Xs(function(){Ol("gel_min_batch_size")>0?Ns().getSequenceCount({cttAuthInfo:void 0,isJspb:a,tier:b})>=Cs&&c():c()},e);
d.i=e}
function Zs(a,b,c,d,e,f,g){e=e===void 0?{}:e;var h=Math.round(W()),k=a.size,l=(g===void 0?0:g)&&U("vss_through_gel_video_stats")?"video_stats":"log_event";a=x(a);var m=a.next();for(g={};!m.done;g={Gc:void 0,batchRequest:void 0,dangerousLogToVisitorSession:void 0,Jc:void 0,Ic:void 0},m=a.next()){var p=x(m.value);m=p.next().value;p=p.next().value;g.batchRequest=Sb({context:yp(b.config_||xp())});if(!Qa(p)&&!U("throw_err_when_logevent_malformed_killswitch")){d();break}g.batchRequest.events=p;(p=Ls[m])&&
bt(g.batchRequest,m,p);delete Ls[m];g.dangerousLogToVisitorSession=m==="visitorOnlyApprovedKey";ct(g.batchRequest,h,g.dangerousLogToVisitorSession);U("always_send_and_write")&&(e.writeThenSend=!1);g.Jc=function(t){U("start_client_gcf")&&cj.pa(function(){return A(function(r){return r.yield(dt(t),0)})});
k--;k||c()};
g.Gc=0;g.Ic=function(t){return function(){t.Gc++;if(e.bypassNetworkless&&t.Gc===1)try{qq(b,l,t.batchRequest,et({writeThenSend:!0},t.dangerousLogToVisitorSession,t.Jc,t.Ic,f)),Ks=!1}catch(r){ul(r),d()}k--;k||c()}}(g);
try{qq(b,l,g.batchRequest,et(e,g.dangerousLogToVisitorSession,g.Jc,g.Ic,f)),Ks=!1}catch(t){ul(t),d()}}}
function et(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,networklessOptions:a,dangerousLogToVisitorSession:b,Eg:!!e,headers:{},postBodyFormat:"",postBody:"",compress:U("compress_gel")||U("compress_gel_lr")};ft()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(W())));return a}
function ct(a,b,c){ft()||(a.requestTimeMs=String(b));U("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=T("EVENT_ID"))&&((c=T("BATCH_CLIENT_COUNTER")||0)||(c=Math.floor(Math.random()*Bs/2)),c++,c>Bs&&(c=1),pl("BATCH_CLIENT_COUNTER",c),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function bt(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function Ps(){var a;(a=E("yt.logging.transport.enableScrapingForTest"))||(a=Nl("il_payload_scraping"),a=(a!==void 0?String(a):"")!=="enable_il_payload_scraping");a||(ts=[],D("yt.logging.transport.enableScrapingForTest",!0),D("yt.logging.transport.scrapedPayloadsForTesting",ts),D("yt.logging.transport.payloadToScrape","visualElementShown visualElementHidden visualElementAttached screenCreated visualElementGestured visualElementStateChanged".split(" ")),D("yt.logging.transport.getScrapedPayloadFromClientEventsFunction"),
D("yt.logging.transport.scrapeClientEvent",!0))}
function ft(){return U("use_request_time_ms_header")||U("lr_use_request_time_ms_header")}
function Xs(a,b){return U("transport_use_scheduler")===!1?Ll(a,b):U("logging_avoid_blocking_during_navigation")||U("lr_logging_avoid_blocking_during_navigation")?Sm(function(){if(ss().currentState==="none")a();else{var c={};ss().install((c.none={callback:a},c))}},b):Sm(a,b)}
function $s(a){U("transport_use_scheduler")?cj.qa(a):window.clearTimeout(a)}
function dt(a){var b,c,d,e,f,g,h,k,l,m;return A(function(p){return p.h==1?(d=(b=a)==null?void 0:(c=b.responseContext)==null?void 0:c.globalConfigGroup,e=ys(d,Vk),g=(f=d)==null?void 0:f.hotHashData,h=ys(d,Uk),l=(k=d)==null?void 0:k.coldHashData,(m=$r().resolve(new Ur(rp)))?g?e?p.yield(tp(m,g,e),2):p.yield(tp(m,g),2):p.B(2):p.return()):l?h?p.yield(up(m,l,h),0):p.yield(up(m,l),0):p.B(0)})}
function Ws(a,b){b=b===void 0?200:b;return a?b===300?Is:Gs:b===300?Hs:Fs}
function Rs(a){a=Object.keys(a);a=x(a);for(var b=a.next();!b.done;b=a.next())if(b=b.value,Mq[b])return b}
function Ss(a){switch(a){case "DELAYED_EVENT_TIER_UNSPECIFIED":return 0;case "DELAYED_EVENT_TIER_DEFAULT":return 100;case "DELAYED_EVENT_TIER_DISPATCH_TO_EMPTY":return 200;case "DELAYED_EVENT_TIER_FAST":return 300;case "DELAYED_EVENT_TIER_IMMEDIATE":return 400;default:return 200}}
;var gt=C.ytLoggingGelSequenceIdObj_||{};D("ytLoggingGelSequenceIdObj_",gt);
function ht(a,b,c,d){d=d===void 0?{}:d;var e={},f=Math.round(d.timestamp||W());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;a=Hr();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};d.sequenceGroup&&!U("web_gel_sequence_info_killswitch")&&(a=e.context,b=d.sequenceGroup,gt[b]=b in gt?gt[b]+1:0,a.sequence={index:gt[b],groupKey:b},d.endOfSequence&&delete gt[d.sequenceGroup]);(d.sendIsolatedPayload?Ts:Os)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,dangerousLogToVisitorSession:d.dangerousLogToVisitorSession},
c)}
;function Hn(a,b,c){c=c===void 0?{}:c;var d=or;T("ytLoggingEventsDefaultDisabled",!1)&&or===or&&(d=null);U("web_all_payloads_via_jspb")&&!c.timestamp&&(c.lact=Hr(),c.timestamp=W());ht(a,b,d,c)}
;D("ytLoggingGelSequenceIdObj_",C.ytLoggingGelSequenceIdObj_||{});var jt=new Set,kt=0,lt=0,mt=0,nt=[],ot=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function Gn(a){pt(a)}
function qt(a){pt(a,"WARNING")}
function rt(a){a instanceof Error?pt(a):(a=Ra(a)?JSON.stringify(a):String(a),a=new V(a),a.name="RejectedPromiseError",qt(a))}
function pt(a,b,c,d,e,f,g,h){f=f===void 0?{}:f;f.name=c||T("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||T("INNERTUBE_CONTEXT_CLIENT_VERSION");c=f;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;if(a&&(a.hasOwnProperty("level")&&a.level&&(b=a.level),U("console_log_js_exceptions")&&(d=[],d.push("Name: "+a.name),d.push("Message: "+a.message),a.hasOwnProperty("params")&&d.push("Error Params: "+JSON.stringify(a.params)),a.hasOwnProperty("args")&&d.push("Error args: "+
JSON.stringify(a.args)),d.push("File name: "+a.fileName),d.push("Stacktrace: "+a.stack),d=d.join("\n"),window.console.log(d,a)),!(kt>=5))){d=nt;var k=fc(a);e=k.message||"Unknown Error";f=k.name||"UnknownError";var l=k.stack||a.i||"Not available";if(l.startsWith(f+": "+e)){var m=l.split("\n");m.shift();l=m.join("\n")}m=k.lineNumber||"Not available";k=k.fileName||"Not available";var p=0;if(a.hasOwnProperty("args")&&a.args&&a.args.length)for(var t=0;t<a.args.length&&!(p=om(a.args[t],"params."+t,c,p),
p>=500);t++);else if(a.hasOwnProperty("params")&&a.params){var r=a.params;if(typeof a.params==="object")for(t in r){if(r[t]){var v="params."+t,u=qm(r[t]);c[v]=u;p+=v.length+u.length;if(p>500)break}}else c.params=qm(r)}if(d.length)for(t=0;t<d.length&&!(p=om(d[t],"params.context."+t,c,p),p>=500);t++);navigator.vendor&&!c.hasOwnProperty("vendor")&&(c["device.vendor"]=navigator.vendor);t={message:e,name:f,lineNumber:m,fileName:k,stack:l,params:c,sampleWeight:1};c=Number(a.columnNumber);isNaN(c)||(t.lineNumber=
t.lineNumber+":"+c);if(a.level==="IGNORED")a=0;else a:{a=km();c=x(a.Va);for(d=c.next();!d.done;d=c.next())if(d=d.value,t.message&&t.message.match(d.Qg)){a=d.weight;break a}a=x(a.Sa);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.callback(t)){a=c.weight;break a}a=1}t.sampleWeight=a;a=x(fm);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.kc[t.name])for(e=x(c.kc[t.name]),d=e.next();!d.done;d=e.next())if(f=d.value,d=t.message.match(f.regexp)){t.params["params.error.original"]=d[0];e=f.groups;f={};
for(m=0;m<e.length;m++)f[e[m]]=d[m+1],t.params["params.error."+e[m]]=d[m+1];t.message=c.Ec(f);break}t.params||(t.params={});a=km();t.params["params.errorServiceSignature"]="msg="+a.Va.length+"&cb="+a.Sa.length;t.params["params.serviceWorker"]="false";C.document&&C.document.querySelectorAll&&(t.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));ib("sample").constructor!==gb&&(t.params["params.fconst"]="true");window.yterr&&typeof window.yterr==="function"&&window.yterr(t);
if(t.sampleWeight!==0&&!jt.has(t.message)){if(g&&U("web_enable_error_204"))st(b===void 0?"ERROR":b,t);else{b=b===void 0?"ERROR":b;b==="ERROR"?(lm.Ya("handleError",t),U("record_app_crashed_web")&&mt===0&&t.sampleWeight===1&&(mt++,g={appCrashType:"APP_CRASH_TYPE_BREAKPAD"},U("report_client_error_with_app_crash_ks")||(g.systemHealth={crashData:{clientError:{logMessage:{message:t.message}}}}),Hn("appCrashed",g)),lt++):b==="WARNING"&&lm.Ya("handleWarning",t);if(U("kevlar_gel_error_routing")){g=b;h=h===
void 0?{}:h;b:{a=x(ot);for(c=a.next();!c.done;c=a.next())if(Nn(c.value.toLowerCase())){a=!0;break b}a=!1}if(a)h=void 0;else{c={stackTrace:t.stack};t.fileName&&(c.filename=t.fileName);a=t.lineNumber&&t.lineNumber.split?t.lineNumber.split(":"):[];a.length!==0&&(a.length!==1||isNaN(Number(a[0]))?a.length!==2||isNaN(Number(a[0]))||isNaN(Number(a[1]))||(c.lineNumber=Number(a[0]),c.columnNumber=Number(a[1])):c.lineNumber=Number(a[0]));a={level:"ERROR_LEVEL_UNKNOWN",message:t.message,errorClassName:t.name,
sampleWeight:t.sampleWeight};g==="ERROR"?a.level="ERROR_LEVEL_ERROR":g==="WARNING"&&(a.level="ERROR_LEVEL_WARNNING");c={isObfuscated:!0,browserStackInfo:c};h.pageUrl=window.location.href;h.kvPairs=[];T("FEXP_EXPERIMENTS")&&(h.experimentIds=T("FEXP_EXPERIMENTS"));d=T("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!ql("web_disable_gel_stp_ecatcher_killswitch")&&d)for(e=x(Object.keys(d)),f=e.next();!f.done;f=e.next())f=f.value,h.kvPairs.push({key:f,value:String(d[f])});if(d=t.params)for(e=x(Object.keys(d)),
f=e.next();!f.done;f=e.next())f=f.value,h.kvPairs.push({key:"client."+f,value:String(d[f])});d=T("SERVER_NAME");e=T("SERVER_VERSION");d&&e&&(h.kvPairs.push({key:"server.name",value:d}),h.kvPairs.push({key:"server.version",value:e}));h={errorMetadata:h,stackTrace:c,logMessage:a}}h&&(Hn("clientError",h),(g==="ERROR"||U("errors_flush_gel_always_killswitch"))&&Vs(void 0,void 0,!1))}U("suppress_error_204_logging")||st(b,t)}try{jt.add(t.message)}catch(z){}kt++}}}
function st(a,b){var c=b.params||{};a={urlParams:{a:"logerror",t:"jserror",type:b.name,msg:b.message.substr(0,250),line:b.lineNumber,level:a,"client.name":c.name},postParams:{url:T("PAGE_NAME",window.location.href),file:b.fileName},method:"POST"};c.version&&(a["client.version"]=c.version);if(a.postParams){b.stack&&(a.postParams.stack=b.stack);b=x(Object.keys(c));for(var d=b.next();!d.done;d=b.next())d=d.value,a.postParams["client."+d]=c[d];if(c=T("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS"))for(b=x(Object.keys(c)),
d=b.next();!d.done;d=b.next())d=d.value,a.postParams[d]=c[d];c=T("SERVER_NAME");b=T("SERVER_VERSION");c&&b&&(a.postParams["server.name"]=c,a.postParams["server.version"]=b)}Xl(T("ECATCHER_REPORT_HOST","")+"/error_204",a)}
;function tt(){this.register=new Map}
function ut(a){a=x(a.register.values());for(var b=a.next();!b.done;b=a.next())b.value.Tg("ABORTED")}
tt.prototype.clear=function(){ut(this);this.register.clear()};
var vt=new tt;var wt=Date.now().toString();
function xt(){a:{if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];var d=a;break a}catch(e){}d=Array(16);for(a=0;a<16;a++){b=Date.now();for(c=0;c<b%23;c++)d[a]=Math.random();d[a]=Math.floor(Math.random()*256)}if(wt)for(a=1,b=0;b<wt.length;b++)d[a%16]=d[a%16]^d[(a-1)%16]/4^wt.charCodeAt(b),a++}a=[];for(b=0;b<d.length;b++)a.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(d[b]&63));
return a.join("")}
;var zt,At=C.ytLoggingDocDocumentNonce_;At||(At=xt(),D("ytLoggingDocDocumentNonce_",At));zt=At;function Bt(a){this.h=a}
n=Bt.prototype;n.getAsJson=function(){var a={};this.h.trackingParams!==void 0?a.trackingParams=this.h.trackingParams:(a.veType=this.h.veType,this.h.veCounter!==void 0&&(a.veCounter=this.h.veCounter),this.h.elementIndex!==void 0&&(a.elementIndex=this.h.elementIndex));this.h.dataElement!==void 0&&(a.dataElement=this.h.dataElement.getAsJson());this.h.youtubeData!==void 0&&(a.youtubeData=this.h.youtubeData);this.h.isCounterfactual&&(a.isCounterfactual=!0);return a};
n.getAsJspb=function(){var a=new bl;this.h.trackingParams!==void 0?a.setTrackingParams(this.h.trackingParams):(this.h.veType!==void 0&&K(a,2,Of(this.h.veType)),this.h.veCounter!==void 0&&K(a,6,Of(this.h.veCounter)),this.h.elementIndex!==void 0&&K(a,3,Of(this.h.elementIndex)),this.h.isCounterfactual&&K(a,5,Kf(!0)));if(this.h.dataElement!==void 0){var b=this.h.dataElement.getAsJspb();Lg(a,bl,7,b)}this.h.youtubeData!==void 0&&Lg(a,Wk,8,this.h.jspbYoutubeData);return a};
n.toString=function(){return JSON.stringify(this.getAsJson())};
n.isClientVe=function(){return!this.h.trackingParams&&!!this.h.veType};
n.getLoggingDirectives=function(){return this.h.loggingDirectives};function Ct(a){return T("client-screen-nonce-store",{})[a===void 0?0:a]}
function Dt(a,b){b=b===void 0?0:b;var c=T("client-screen-nonce-store");c||(c={},pl("client-screen-nonce-store",c));c[b]=a}
function Et(a){a=a===void 0?0:a;return a===0?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function Ft(a){return T(Et(a===void 0?0:a))}
D("yt_logging_screen.getRootVeType",Ft);function Gt(){var a=T("csn-to-ctt-auth-info");a||(a={},pl("csn-to-ctt-auth-info",a));return a}
function Ht(){return Object.values(T("client-screen-nonce-store",{})).filter(function(a){return a!==void 0})}
function It(a){a=Ct(a===void 0?0:a);if(!a&&!T("USE_CSN_FALLBACK",!0))return null;a||(a="UNDEFINED_CSN");return a?a:null}
D("yt_logging_screen.getCurrentCsn",It);function Jt(a,b,c){var d=Gt();(c=It(c))&&delete d[c];b&&(d[a]=b)}
function Kt(a){return Gt()[a]}
D("yt_logging_screen.getCttAuthInfo",Kt);D("yt_logging_screen.setCurrentScreen",function(a,b,c,d){c=c===void 0?0:c;if(a!==Ct(c)||b!==T(Et(c)))if(Jt(a,d,c),Dt(a,c),pl(Et(c),b),b=function(){setTimeout(function(){a&&Hn("foregroundHeartbeatScreenAssociated",{clientDocumentNonce:zt,clientScreenNonce:a})},0)},"requestAnimationFrame"in window)try{window.requestAnimationFrame(b)}catch(e){b()}else b()});function Lt(){var a=Rb(Mt),b;return(new Ud(function(c,d){a.onSuccess=function(e){Kl(e)?c(new Nt(e)):d(new Ot("Request failed, status="+(e&&"status"in e?e.status:-1),"net.badstatus",e))};
a.onError=function(e){d(new Ot("Unknown request error","net.unknown",e))};
a.onTimeout=function(e){d(new Ot("Request timed out","net.timeout",e))};
b=Xl("//googleads.g.doubleclick.net/pagead/id",a)})).qc(function(c){if(c instanceof ae){var d;
(d=b)==null||d.abort()}return Zd(c)})}
function Ot(a,b,c){bb.call(this,a+", errorCode="+b);this.errorCode=b;this.xhr=c;this.name="PromiseAjaxError"}
y(Ot,bb);function Nt(a){this.xhr=a}
;function Pt(){this.h=0;this.i=null}
Pt.prototype.then=function(a,b,c){return this.h===1&&a?(a=a.call(c,this.i))&&typeof a.then==="function"?a:Qt(a):this.h===2&&b?(a=b.call(c,this.i))&&typeof a.then==="function"?a:Rt(a):this};
Pt.prototype.getValue=function(){return this.i};
Pt.prototype.isRejected=function(){return this.h==2};
Pt.prototype.$goog_Thenable=!0;function Rt(a){var b=new Pt;a=a===void 0?null:a;b.h=2;b.i=a===void 0?null:a;return b}
function Qt(a){var b=new Pt;a=a===void 0?null:a;b.h=1;b.i=a===void 0?null:a;return b}
;function St(a,b){var c=c===void 0?{}:c;a={method:b===void 0?"POST":b,mode:Fl(a)?"same-origin":"cors",credentials:Fl(a)?"same-origin":"include"};b={};for(var d=x(Object.keys(c)),e=d.next();!e.done;e=d.next())e=e.value,c[e]&&(b[e]=c[e]);Object.keys(b).length>0&&(a.headers=b);return a}
;function Tt(){return Ah()||(De||Ee)&&Nn("applewebkit")&&!Nn("version")&&(!Nn("safari")||Nn("gsa/"))||Zc&&Nn("version/")?!0:T("EOM_VISITOR_DATA")?!1:!0}
;function Ut(a){a:{var b="EMBEDDED_PLAYER_MODE_UNKNOWN";window.location.hostname.includes("youtubeeducation.com")&&(b="EMBEDDED_PLAYER_MODE_PFL");var c=a.raw_embedded_player_response;if(!c&&(a=a.embedded_player_response))try{c=JSON.parse(a)}catch(e){break a}if(c)b:for(var d in $k)if($k[d]==c.embeddedPlayerMode){b=$k[d];break b}}return b==="EMBEDDED_PLAYER_MODE_PFL"}
;function Vt(a){bb.call(this,a.message||a.description||a.name);this.isMissing=a instanceof Wt;this.isTimeout=a instanceof Ot&&a.errorCode=="net.timeout";this.isCanceled=a instanceof ae}
y(Vt,bb);Vt.prototype.name="BiscottiError";function Wt(){bb.call(this,"Biscotti ID is missing from server")}
y(Wt,bb);Wt.prototype.name="BiscottiMissingError";var Mt={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},Xt=null;function Yt(){if(U("disable_biscotti_fetch_entirely_for_all_web_clients"))return Error("Biscotti id fetching has been disabled entirely.");if(!Tt())return Error("User has not consented - not fetching biscotti id.");var a=T("PLAYER_VARS",{});if(Pb(a)=="1")return Error("Biscotti ID is not available in private embed mode");if(Ut(a))return Error("Biscotti id fetching has been disabled for pfl.")}
function il(){var a=Yt();if(a!==void 0)return Zd(a);Xt||(Xt=Lt().then(Zt).qc(function(b){return $t(2,b)}));
return Xt}
function Zt(a){a=a.xhr.responseText;if(a.lastIndexOf(")]}'",0)!=0)throw new Wt;a=JSON.parse(a.substr(4));if((a.type||1)>1)throw new Wt;a=a.id;jl(a);Xt=Qt(a);au(18E5,2);return a}
function $t(a,b){b=new Vt(b);jl("");Xt=Rt(b);a>0&&au(12E4,a-1);throw b;}
function au(a,b){Ll(function(){Lt().then(Zt,function(c){return $t(b,c)}).qc(Bd)},a)}
function bu(){try{var a=E("yt.ads.biscotti.getId_");return a?a():il()}catch(b){return Zd(b)}}
;var Bb=ja(["data-"]);function cu(a){a&&(a.dataset?a.dataset[du()]="true":Wb(a))}
function eu(a){return a?a.dataset?a.dataset[du()]:a.getAttribute("data-loaded"):null}
var fu={};function du(){return fu.loaded||(fu.loaded="loaded".replace(/\-([a-z])/g,function(a,b){return b.toUpperCase()}))}
;function gu(a){a=a||{};var b={},c={};this.url=a.url||"";this.args=a.args||Rb(b);this.assets=a.assets||{};this.attrs=a.attrs||Rb(c);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
gu.prototype.clone=function(){var a=new gu,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];Pa(c)=="object"?a[b]=Rb(c):a[b]=c}return a};var hu=["share/get_web_player_share_panel"],iu=["feedback"],ju=["notification/modify_channel_preference"],ku=["browse/edit_playlist"],lu=["subscription/subscribe"],mu=["subscription/unsubscribe"];var nu=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};D("yt.msgs_",nu);function ou(a){kl(nu,arguments)}
;function pu(a,b,c){qu(a,b,c===void 0?null:c)}
function ru(a){a=su(a);var b=document.getElementById(a);b&&(Qr(a),b.parentNode.removeChild(b))}
function tu(a,b){a&&b&&(a=""+Sa(b),(a=uu[a])&&Or(a))}
function qu(a,b,c){c=c===void 0?null:c;var d=su(a),e=document.getElementById(d),f=e&&eu(e),g=e&&!f;f?b&&b():(b&&(f=Mr(d,b),b=""+Sa(b),uu[b]=f),g||(e=vu(a,d,function(){eu(e)||(cu(e),Pr(d),Ll(function(){Qr(d)},0))},c)))}
function vu(a,b,c,d){d=d===void 0?null:d;var e=Gd("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);dc(e,Sk(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function su(a){var b=document.createElement("a");zb(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+kc(a)}
var uu={};function wu(a){var b=xu(a),c=document.getElementById(b),d=c&&eu(c);d||c&&!d||(c=yu(a,b,function(){if(!eu(c)){cu(c);Pr(b);var e=Ya(Qr,b);Ll(e,0)}}))}
function yu(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Sk(a);Zb(d,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function xu(a){var b=Gd("A");zb(b,new sb(a));a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+kc(a)}
;function zu(a){var b=B.apply(1,arguments);if(!Au(a)||b.some(function(d){return!Au(d)}))throw Error("Only objects may be merged.");
b=x(b);for(var c=b.next();!c.done;c=b.next())Bu(a,c.value)}
function Bu(a,b){for(var c in b)if(Au(b[c])){if(c in a&&!Au(a[c]))throw Error("Cannot merge an object into a non-object.");c in a||(a[c]={});Bu(a[c],b[c])}else if(Cu(b[c])){if(c in a&&!Cu(a[c]))throw Error("Cannot merge an array into a non-array.");c in a||(a[c]=[]);Du(a[c],b[c])}else a[c]=b[c];return a}
function Du(a,b){b=x(b);for(var c=b.next();!c.done;c=b.next())c=c.value,Au(c)?a.push(Bu({},c)):Cu(c)?a.push(Du([],c)):a.push(c);return a}
function Au(a){return typeof a==="object"&&!Array.isArray(a)}
function Cu(a){return typeof a==="object"&&Array.isArray(a)}
;function Eu(a){a=a===void 0?!1:a;G.call(this);this.h=new M(a);Ec(this,this.h)}
$a(Eu,G);Eu.prototype.subscribe=function(a,b,c){return this.V?0:this.h.subscribe(a,b,c)};
Eu.prototype.unsubscribe=function(a,b,c){return this.V?!1:this.h.unsubscribe(a,b,c)};
Eu.prototype.l=function(a,b){this.V||this.h.Ya.apply(this.h,arguments)};var Fu="absolute_experiments app conditional_experiments debugcss debugjs expflag forced_experiments pbj pbjreload sbb spf spfreload sr_bns_address sttick".split(" ");
function Gu(a,b){var c=c===void 0?!0:c;var d=T("VALID_SESSION_TEMPDATA_DOMAINS",[]),e=oc(window.location.href);e&&d.push(e);e=oc(a);if(Cb(d,e)>=0||!e&&a.lastIndexOf("/",0)==0)if(d=document.createElement("a"),zb(d,a),a=d.href)if(a=pc(a),a=qc(a))if(c&&!b.csn&&(b.itct||b.ved)&&(b=Object.assign({csn:It()},b)),f){var f=parseInt(f,10);isFinite(f)&&f>0&&Hu(a,b,f)}else Hu(a,b)}
function Hu(a,b,c){a=Iu(a);b=b?sc(b):"";c=c||5;Tt()&&ym(a,b,c)}
function Iu(a){for(var b=x(Fu),c=b.next();!c.done;c=b.next())a=xc(a,c.value);return"ST-"+kc(a).toString(36)}
;function Ju(a){Cp.call(this,1,arguments);this.csn=a}
y(Ju,Cp);var Lp=new Dp("screen-created",Ju),Ku=[],Lu=0,Mu=new Map,Nu=new Map,Ou=new Map;
function Pu(a,b,c,d,e){e=e===void 0?!1:e;for(var f=Qu({cttAuthInfo:Kt(b)||void 0},b),g=x(d),h=g.next();!h.done;h=g.next()){h=h.value;var k=h.getAsJson();(Nb(k)||!k.trackingParams&&!k.veType)&&qt(Error("Child VE logged with no data"));if(U("no_client_ve_attach_unless_shown")){var l=Ru(h,b);if(k.veType&&!Nu.has(l)&&!Ou.has(l)&&!e){if(!U("il_attach_cache_limit")||Mu.size<1E3){Mu.set(l,[a,b,c,h]);return}U("il_attach_cache_limit")&&Mu.size>1E3&&qt(new V("IL Attach cache exceeded limit"))}h=Ru(c,b);Mu.has(h)?
Su(c,b):Ou.set(h,!0)}}d=d.filter(function(m){m.csn!==b?(m.csn=b,m=!0):m=!1;return m});
c={csn:b,parentVe:c.getAsJson(),childVes:Fb(d,function(m){return m.getAsJson()})};
b==="UNDEFINED_CSN"?Tu("visualElementAttached",f,c):a?ht("visualElementAttached",c,a,f):Hn("visualElementAttached",c,f)}
function Tu(a,b,c){Ku.push({Be:a,payload:c,Mg:void 0,options:b});Lu||(Lu=Mp())}
function Np(a){if(Ku){for(var b=x(Ku),c=b.next();!c.done;c=b.next())c=c.value,c.payload&&(c.payload.csn=a.csn,Hn(c.Be,c.payload,c.options));Ku.length=0}Lu=0}
function Ru(a,b){return""+a.getAsJson().veType+a.getAsJson().veCounter+b}
function Su(a,b){a=Ru(a,b);Mu.has(a)&&(b=Mu.get(a)||[],Pu(b[0],b[1],b[2],[b[3]],!0),Mu.delete(a))}
function Qu(a,b){U("log_sequence_info_on_gel_web")&&(a.sequenceGroup=b);return a}
;function Uu(){try{return!!self.localStorage}catch(a){return!1}}
;function Vu(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function Wu(a){if(Uu()){var b=Object.keys(window.localStorage);b=x(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=Vu(c);d===void 0||a.includes(d)||self.localStorage.removeItem(c)}}}
function Xu(){if(!Uu())return!1;var a=Qm(),b=Object.keys(window.localStorage);b=x(b);for(var c=b.next();!c.done;c=b.next())if(c=Vu(c.value),c!==void 0&&c!==a)return!0;return!1}
;function Yu(){var a=!1;try{a=!!window.sessionStorage.getItem("session_logininfo")}catch(b){a=!0}return(T("INNERTUBE_CLIENT_NAME")==="WEB"||T("INNERTUBE_CLIENT_NAME")==="WEB_CREATOR")&&a}
function Zu(a){if(T("LOGGED_IN",!0)&&Yu()){var b=T("VALID_SESSION_TEMPDATA_DOMAINS",[]);var c=oc(window.location.href);c&&b.push(c);c=oc(a);Cb(b,c)>=0||!c&&a.lastIndexOf("/",0)==0?(b=pc(a),(b=qc(b))?(b=Iu(b),b=(b=zm(b)||null)?Cl(b):{}):b=null):b=null;b==null&&(b={});c=b;var d=void 0;Yu()?(d||(d=T("LOGIN_INFO")),d?(c.session_logininfo=d,c=!0):c=!1):c=!1;c&&Gu(a,b)}}
;function $u(a,b,c){b=b===void 0?{}:b;c=c===void 0?!1:c;var d=T("EVENT_ID");d&&(b.ei||(b.ei=d));b&&Gu(a,b);if(c)return!1;Zu(a);var e=e===void 0?{}:e;var f=f===void 0?"":f;var g=g===void 0?window:g;a=tc(a,e);Zu(a);f=a+f;var h=h===void 0?wb:h;a:if(h=h===void 0?wb:h,f instanceof sb)h=f;else{for(a=0;a<h.length;++a)if(b=h[a],b instanceof ub&&b.pe(f)){h=new sb(f);break a}h=void 0}g=g.location;h=yb(h||tb);h!==void 0&&(g.href=h);return!0}
;function av(a){if(Pb(T("PLAYER_VARS",{}))!="1"){a&&hl();try{bu().then(function(){},function(){}),Ll(av,18E5)}catch(b){ul(b)}}}
;var bv=new Map([["dark","USER_INTERFACE_THEME_DARK"],["light","USER_INTERFACE_THEME_LIGHT"]]);function cv(){var a=a===void 0?window.location.href:a;if(U("kevlar_disable_theme_param"))return null;mc(nc(5,a));try{var b=Dl(a).theme;return bv.get(b)||null}catch(c){}return null}
;function dv(){this.h={};if(this.i=Bm()){var a=zm("CONSISTENCY");a&&ev(this,{encryptedTokenJarContents:a})}}
dv.prototype.handleResponse=function(a,b){if(!b)throw Error("request needs to be passed into ConsistencyService");var c,d;b=((c=b.Oa.context)==null?void 0:(d=c.request)==null?void 0:d.consistencyTokenJars)||[];var e;if(a=(e=a.responseContext)==null?void 0:e.consistencyTokenJar){e=x(b);for(c=e.next();!c.done;c=e.next())delete this.h[c.value.encryptedTokenJarContents];ev(this,a)}};
function ev(a,b){if(b.encryptedTokenJarContents&&(a.h[b.encryptedTokenJarContents]=b,typeof b.expirationSeconds==="string")){var c=Number(b.expirationSeconds);setTimeout(function(){delete a.h[b.encryptedTokenJarContents]},c*1E3);
a.i&&ym("CONSISTENCY",b.encryptedTokenJarContents,c,void 0,!0)}}
;var fv=window.location.hostname.split(".").slice(-2).join(".");function gv(){this.j=-1;var a=T("LOCATION_PLAYABILITY_TOKEN");T("INNERTUBE_CLIENT_NAME")==="TVHTML5"&&(this.h=hv(this))&&(a=this.h.get("yt-location-playability-token"));a&&(this.locationPlayabilityToken=a,this.i=void 0)}
var iv;function jv(){iv=E("yt.clientLocationService.instance");iv||(iv=new gv,D("yt.clientLocationService.instance",iv));return iv}
n=gv.prototype;
n.setLocationOnInnerTubeContext=function(a){a.client||(a.client={});if(this.i)a.client.locationInfo||(a.client.locationInfo={}),a.client.locationInfo.latitudeE7=Math.floor(this.i.coords.latitude*1E7),a.client.locationInfo.longitudeE7=Math.floor(this.i.coords.longitude*1E7),a.client.locationInfo.horizontalAccuracyMeters=Math.round(this.i.coords.accuracy),a.client.locationInfo.forceLocationPlayabilityTokenRefresh=!0;else if(this.l||this.locationPlayabilityToken)a.client.locationPlayabilityToken=this.l||
this.locationPlayabilityToken};
n.handleResponse=function(a){var b;a=(b=a.responseContext)==null?void 0:b.locationPlayabilityToken;a!==void 0&&(this.locationPlayabilityToken=a,this.i=void 0,T("INNERTUBE_CLIENT_NAME")==="TVHTML5"?(this.h=hv(this))&&this.h.set("yt-location-playability-token",a,15552E3):ym("YT_CL",JSON.stringify({loctok:a}),15552E3,fv,!0))};
function hv(a){return a.h===void 0?new yn("yt-client-location"):a.h}
n.clearLocationPlayabilityToken=function(a){a==="TVHTML5"?(this.h=hv(this))&&this.h.remove("yt-location-playability-token"):Am("YT_CL");this.l=void 0;this.j!==-1&&(clearTimeout(this.j),this.j=-1)};
n.getCurrentPositionFromGeolocation=function(){var a=this;if(!(navigator&&navigator.geolocation&&navigator.geolocation.getCurrentPosition))return Promise.reject(Error("Geolocation unsupported"));var b=!1,c=1E4;T("INNERTUBE_CLIENT_NAME")==="MWEB"&&(b=!0,c=15E3);return new Promise(function(d,e){navigator.geolocation.getCurrentPosition(function(f){a.i=f;d(f)},function(f){e(f)},{enableHighAccuracy:b,
maximumAge:0,timeout:c})})};
n.createUnpluggedLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);if(a==null?0:a.accuracy)b.locationRadiusMeters=Math.round(a.accuracy);return b};
n.createLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);return b};function kv(a){var b={"Content-Type":"application/json"};T("EOM_VISITOR_DATA")?b["X-Goog-EOM-Visitor-Id"]=T("EOM_VISITOR_DATA"):T("VISITOR_DATA")&&(b["X-Goog-Visitor-Id"]=T("VISITOR_DATA"));b["X-Youtube-Bootstrap-Logged-In"]=T("LOGGED_IN",!1);T("DEBUG_SETTINGS_METADATA")&&(b["X-Debug-Settings-Metadata"]=T("DEBUG_SETTINGS_METADATA"));a!=="cors"&&((a=T("INNERTUBE_CONTEXT_CLIENT_NAME"))&&(b["X-Youtube-Client-Name"]=a),(a=T("INNERTUBE_CONTEXT_CLIENT_VERSION"))&&(b["X-Youtube-Client-Version"]=a),(a=T("CHROME_CONNECTED_HEADER"))&&
(b["X-Youtube-Chrome-Connected"]=a),(a=T("DOMAIN_ADMIN_STATE"))&&(b["X-Youtube-Domain-Admin-State"]=a));return b}
;function lv(){this.h={}}
lv.prototype.contains=function(a){return Object.prototype.hasOwnProperty.call(this.h,a)};
lv.prototype.get=function(a){if(this.contains(a))return this.h[a]};
lv.prototype.set=function(a,b){this.h[a]=b};
lv.prototype.remove=function(a){delete this.h[a]};function mv(){this.mappings=new lv}
mv.prototype.getModuleId=function(a){return a.serviceId.getModuleId()};
mv.prototype.get=function(a){a:{var b=this.mappings.get(a.toString());switch(b.type){case "mapping":a=b.value;break a;case "factory":b=b.value();this.mappings.set(a.toString(),{type:"mapping",value:b});a=b;break a;default:a=Xb(b)}}return a};
new mv;function nv(a){return function(){return new a}}
;var ov={},pv=(ov.WEB_UNPLUGGED="^unplugged/",ov.WEB_UNPLUGGED_ONBOARDING="^unplugged/",ov.WEB_UNPLUGGED_OPS="^unplugged/",ov.WEB_UNPLUGGED_PUBLIC="^unplugged/",ov.WEB_CREATOR="^creator/",ov.WEB_KIDS="^kids/",ov.WEB_EXPERIMENTS="^experiments/",ov.WEB_MUSIC="^music/",ov.WEB_REMIX="^music/",ov.WEB_MUSIC_EMBEDDED_PLAYER="^music/",ov.WEB_MUSIC_EMBEDDED_PLAYER="^main_app/|^sfv/",ov);
function qv(a){var b=b===void 0?"UNKNOWN_INTERFACE":b;if(a.length===1)return a[0];var c=pv[b];if(c){c=new RegExp(c);for(var d=x(a),e=d.next();!e.done;e=d.next())if(e=e.value,c.exec(e))return e}var f=[];Object.entries(pv).forEach(function(g){var h=x(g);g=h.next().value;h=h.next().value;b!==g&&f.push(h)});
c=new RegExp(f.join("|"));a.sort(function(g,h){return g.length-h.length});
d=x(a);for(e=d.next();!e.done;e=d.next())if(e=e.value,!c.exec(e))return e;return a[0]}
;function rv(){}
rv.prototype.v=function(a,b,c){b=b===void 0?{}:b;c=c===void 0?xm:c;var d=a.clickTrackingParams,e=this.l,f=!1;f=f===void 0?!1:f;e=e===void 0?!1:e;var g=T("INNERTUBE_CONTEXT");if(g){g=Sb(g);U("web_no_tracking_params_in_shell_killswitch")||delete g.clickTracking;g.client||(g.client={});var h=g.client;h.clientName==="MWEB"&&h.clientFormFactor!=="AUTOMOTIVE_FORM_FACTOR"&&(h.clientFormFactor=T("IS_TABLET")?"LARGE_FORM_FACTOR":"SMALL_FORM_FACTOR");h.screenWidthPoints=window.innerWidth;h.screenHeightPoints=
window.innerHeight;h.screenPixelDensity=Math.round(window.devicePixelRatio||1);h.screenDensityFloat=window.devicePixelRatio||1;h.utcOffsetMinutes=-Math.floor((new Date).getTimezoneOffset());var k=k===void 0?!1:k;Fm();var l="USER_INTERFACE_THEME_LIGHT";Im(165)?l="USER_INTERFACE_THEME_DARK":Im(174)?l="USER_INTERFACE_THEME_LIGHT":!U("kevlar_legacy_browsers")&&window.matchMedia&&window.matchMedia("(prefers-color-scheme)").matches&&window.matchMedia("(prefers-color-scheme: dark)").matches&&(l="USER_INTERFACE_THEME_DARK");
k=k?l:cv()||l;h.userInterfaceTheme=k;if(!f){if(k=Nm())h.connectionType=k;U("web_log_effective_connection_type")&&(k=Om())&&(g.client.effectiveConnectionType=k)}var m;if(U("web_log_memory_total_kbytes")&&((m=C.navigator)==null?0:m.deviceMemory)){var p;m=(p=C.navigator)==null?void 0:p.deviceMemory;g.client.memoryTotalKbytes=""+m*1E6}U("web_gcf_hashes_innertube")&&(k=vp())&&(p=k.coldConfigData,m=k.coldHashData,k=k.hotHashData,g.client.configInfo=g.client.configInfo||{},p&&(g.client.configInfo.coldConfigData=
p),m&&(g.client.configInfo.coldHashData=m),k&&(g.client.configInfo.hotHashData=k));p=Dl(C.location.href);!U("web_populate_internal_geo_killswitch")&&p.internalcountrycode&&(h.internalGeo=p.internalcountrycode);h.clientName==="MWEB"||h.clientName==="WEB"?(h.mainAppWebInfo={graftUrl:C.location.href},U("kevlar_woffle")&&rm.h&&(p=rm.h,h.mainAppWebInfo.pwaInstallabilityStatus=!p.h&&p.i?"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED":"PWA_INSTALLABILITY_STATUS_UNKNOWN"),h.mainAppWebInfo.webDisplayMode=sm(),
h.mainAppWebInfo.isWebNativeShareAvailable=navigator&&navigator.share!==void 0):h.clientName==="TVHTML5"&&(!U("web_lr_app_quality_killswitch")&&(p=T("LIVING_ROOM_APP_QUALITY"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||{},{appQuality:p})),p=T("LIVING_ROOM_CERTIFICATION_SCOPE"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||{},{certificationScope:p}));if(!U("web_populate_time_zone_itc_killswitch")){b:{if(typeof Intl!=="undefined")try{var t=(new Intl.DateTimeFormat).resolvedOptions().timeZone;break b}catch(ea){}t=
void 0}t&&(h.timeZone=t)}(t=T("EXPERIMENTS_TOKEN",""))?h.experimentsToken=t:delete h.experimentsToken;t=Pl();dv.h||(dv.h=new dv);h=dv.h.h;p=[];m=0;for(var r in h)p[m++]=h[r];g.request=Object.assign({},g.request,{internalExperimentFlags:t,consistencyTokenJars:p});!U("web_prequest_context_killswitch")&&(r=T("INNERTUBE_CONTEXT_PREQUEST_CONTEXT"))&&(g.request.externalPrequestContext=r);t=Fm();r=Im(58);t=t.get("gsml","");g.user=Object.assign({},g.user);r&&(g.user.enableSafetyMode=r);t&&(g.user.lockedSafetyMode=
!0);U("warm_op_csn_cleanup")?e&&(f=It())&&(g.clientScreenNonce=f):!f&&(f=It())&&(g.clientScreenNonce=f);d&&(g.clickTracking={clickTrackingParams:d});if(d=E("yt.mdx.remote.remoteClient_"))g.remoteClient=d;jv().setLocationOnInnerTubeContext(g);try{var v=Gl(),u=v.bid;delete v.bid;g.adSignalsInfo={params:[],bid:u};var z=x(Object.entries(v));for(var F=z.next();!F.done;F=z.next()){var J=x(F.value),O=J.next().value,S=J.next().value;v=O;u=S;d=void 0;(d=g.adSignalsInfo.params)==null||d.push({key:v,value:""+
u})}var da;if(U("add_ifa_to_tvh5_requests")&&((da=g.client)==null?void 0:da.clientName)==="TVHTML5"){var va=T("INNERTUBE_CONTEXT");va.adSignalsInfo&&(g.adSignalsInfo.advertisingId=va.adSignalsInfo.advertisingId,g.adSignalsInfo.advertisingIdSignalType="DEVICE_ID_TYPE_CONNECTED_TV_IFA",g.adSignalsInfo.limitAdTracking=va.adSignalsInfo.limitAdTracking)}}catch(ea){pt(ea)}z=g}else pt(Error("Error: No InnerTubeContext shell provided in ytconfig.")),z={};z={context:z};if(F=this.i(a)){this.h(z,F,b);var P;
b="/youtubei/v1/"+qv(this.j());(F=(P=ys(a.commandMetadata,Yk))==null?void 0:P.apiUrl)&&(b=F);P=b;(b=T("INNERTUBE_HOST_OVERRIDE"))&&(P=String(b)+String(pc(P)));b={};U("web_api_key_killswitch")&&(b.key=T("INNERTUBE_API_KEY"));U("json_condensed_response")&&(b.prettyPrint="false");P=El(P,b||{},!1);a=Object.assign({},{command:a},void 0);a={input:P,ib:St(P),Oa:z,config:a};a.config.Wb?a.config.Wb.identity=c:a.config.Wb={identity:c};return a}pt(new V("Error: Failed to create Request from Command.",a))};
fa.Object.defineProperties(rv.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!1}}});
function sv(){}
y(sv,rv);function tv(){}
y(tv,sv);tv.prototype.v=function(){return{input:"/getDatasyncIdsEndpoint",ib:St("/getDatasyncIdsEndpoint","GET"),Oa:{}}};
tv.prototype.j=function(){return[]};
tv.prototype.i=function(){};
tv.prototype.h=function(){};var uv={},vv=(uv.GET_DATASYNC_IDS=nv(tv),uv);function wv(a){var b;(b=E("ytcsi."+(a||"")+"data_"))||(b={tick:{},info:{}},D("ytcsi."+(a||"")+"data_",b));return b}
function xv(){var a=wv();a.info||(a.info={});return a.info}
function yv(a){a=wv(a);a.metadata||(a.metadata={});return a.metadata}
function zv(a){a=wv(a);a.tick||(a.tick={});return a.tick}
function Av(a){a=wv(a);if(a.gel){var b=a.gel;b.gelInfos||(b.gelInfos={});b.gelTicks||(b.gelTicks={})}else a.gel={gelTicks:{},gelInfos:{}};return a.gel}
function Bv(a){a=Av(a);a.gelInfos||(a.gelInfos={});return a.gelInfos}
function Cv(a){var b=wv(a).nonce;b||(b=xt(),wv(a).nonce=b);return b}
;function Dv(){var a=E("ytcsi.debug");a||(a=[],D("ytcsi.debug",a),D("ytcsi.reference",{}));return a}
function Ev(a){a=a||"";var b=E("ytcsi.reference");b||(Dv(),b=E("ytcsi.reference"));if(b[a])return b[a];var c=Dv(),d={timerName:a,info:{},tick:{},span:{},jspbInfo:[]};c.push(d);return b[a]=d}
;var X={},Fv=(X.auto_search="LATENCY_ACTION_AUTO_SEARCH",X.ad_to_ad="LATENCY_ACTION_AD_TO_AD",X.ad_to_video="LATENCY_ACTION_AD_TO_VIDEO",X["analytics.explore"]="LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE",X.app_startup="LATENCY_ACTION_APP_STARTUP",X["artist.analytics"]="LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS",X["artist.events"]="LATENCY_ACTION_CREATOR_ARTIST_CONCERTS",X["artist.presskit"]="LATENCY_ACTION_CREATOR_ARTIST_PROFILE",X["asset.claimed_videos"]="LATENCY_ACTION_CREATOR_CMS_ASSET_CLAIMED_VIDEOS",
X["asset.composition"]="LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION",X["asset.composition_ownership"]="LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION_OWNERSHIP",X["asset.composition_policy"]="LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION_POLICY",X["asset.embeds"]="LATENCY_ACTION_CREATOR_CMS_ASSET_EMBEDS",X["asset.history"]="LATENCY_ACTION_CREATOR_CMS_ASSET_HISTORY",X["asset.issues"]="LATENCY_ACTION_CREATOR_CMS_ASSET_ISSUES",X["asset.licenses"]="LATENCY_ACTION_CREATOR_CMS_ASSET_LICENSES",X["asset.metadata"]=
"LATENCY_ACTION_CREATOR_CMS_ASSET_METADATA",X["asset.ownership"]="LATENCY_ACTION_CREATOR_CMS_ASSET_OWNERSHIP",X["asset.policy"]="LATENCY_ACTION_CREATOR_CMS_ASSET_POLICY",X["asset.references"]="LATENCY_ACTION_CREATOR_CMS_ASSET_REFERENCES",X["asset.shares"]="LATENCY_ACTION_CREATOR_CMS_ASSET_SHARES",X["asset.sound_recordings"]="LATENCY_ACTION_CREATOR_CMS_ASSET_SOUND_RECORDINGS",X["asset_group.assets"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUP_ASSETS",X["asset_group.campaigns"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUP_CAMPAIGNS",
X["asset_group.claimed_videos"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUP_CLAIMED_VIDEOS",X["asset_group.metadata"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUP_METADATA",X["song.analytics"]="LATENCY_ACTION_CREATOR_SONG_ANALYTICS",X.browse="LATENCY_ACTION_BROWSE",X.cast_splash="LATENCY_ACTION_CAST_SPLASH",X.channels="LATENCY_ACTION_CHANNELS",X.creator_channel_dashboard="LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD",X["channel.analytics"]="LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS",X["channel.comments"]="LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS",
X["channel.content"]="LATENCY_ACTION_CREATOR_POST_LIST",X["channel.content.promotions"]="LATENCY_ACTION_CREATOR_PROMOTION_LIST",X["channel.copyright"]="LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT",X["channel.editing"]="LATENCY_ACTION_CREATOR_CHANNEL_EDITING",X["channel.monetization"]="LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION",X["channel.music"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC",X["channel.music_storefront"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT",X["channel.playlists"]="LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS",
X["channel.translations"]="LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS",X["channel.videos"]="LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS",X["channel.live_streaming"]="LATENCY_ACTION_CREATOR_LIVE_STREAMING",X.chips="LATENCY_ACTION_CHIPS",X.commerce_transaction="LATENCY_ACTION_COMMERCE_TRANSACTION",X["dialog.copyright_strikes"]="LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES",X["dialog.video_copyright"]="LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT",X["dialog.uploads"]="LATENCY_ACTION_CREATOR_DIALOG_UPLOADS",
X.direct_playback="LATENCY_ACTION_DIRECT_PLAYBACK",X.embed="LATENCY_ACTION_EMBED",X.entity_key_serialization_perf="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF",X.entity_key_deserialization_perf="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF",X.explore="LATENCY_ACTION_EXPLORE",X.favorites="LATENCY_ACTION_FAVORITES",X.home="LATENCY_ACTION_HOME",X.inboarding="LATENCY_ACTION_INBOARDING",X.library="LATENCY_ACTION_LIBRARY",X.live="LATENCY_ACTION_LIVE",X.live_pagination="LATENCY_ACTION_LIVE_PAGINATION",
X.mini_app="LATENCY_ACTION_MINI_APP_PLAY",X.notification_settings="LATENCY_ACTION_KIDS_NOTIFICATION_SETTINGS",X.onboarding="LATENCY_ACTION_ONBOARDING",X.owner="LATENCY_ACTION_CREATOR_CMS_DASHBOARD",X["owner.allowlist"]="LATENCY_ACTION_CREATOR_CMS_ALLOWLIST",X["owner.analytics"]="LATENCY_ACTION_CREATOR_CMS_ANALYTICS",X["owner.art_tracks"]="LATENCY_ACTION_CREATOR_CMS_ART_TRACKS",X["owner.assets"]="LATENCY_ACTION_CREATOR_CMS_ASSETS",X["owner.asset_groups"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUPS",X["owner.bulk"]=
"LATENCY_ACTION_CREATOR_CMS_BULK_HISTORY",X["owner.campaigns"]="LATENCY_ACTION_CREATOR_CMS_CAMPAIGNS",X["owner.channel_invites"]="LATENCY_ACTION_CREATOR_CMS_CHANNEL_INVITES",X["owner.channels"]="LATENCY_ACTION_CREATOR_CMS_CHANNELS",X["owner.claimed_videos"]="LATENCY_ACTION_CREATOR_CMS_CLAIMED_VIDEOS",X["owner.claims"]="LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING",X["owner.claims.manual"]="LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING",X["owner.delivery"]="LATENCY_ACTION_CREATOR_CMS_CONTENT_DELIVERY",
X["owner.delivery_templates"]="LATENCY_ACTION_CREATOR_CMS_DELIVERY_TEMPLATES",X["owner.issues"]="LATENCY_ACTION_CREATOR_CMS_ISSUES",X["owner.licenses"]="LATENCY_ACTION_CREATOR_CMS_LICENSES",X["owner.pitch_music"]="LATENCY_ACTION_CREATOR_CMS_PITCH_MUSIC",X["owner.policies"]="LATENCY_ACTION_CREATOR_CMS_POLICIES",X["owner.releases"]="LATENCY_ACTION_CREATOR_CMS_RELEASES",X["owner.reports"]="LATENCY_ACTION_CREATOR_CMS_REPORTS",X["owner.videos"]="LATENCY_ACTION_CREATOR_CMS_VIDEOS",X.parent_profile_settings=
"LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS",X.parent_tools_collection="LATENCY_ACTION_PARENT_TOOLS_COLLECTION",X.parent_tools_dashboard="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD",X.player_att="LATENCY_ACTION_PLAYER_ATTESTATION",X["playlist.videos"]="LATENCY_ACTION_CREATOR_PLAYLIST_VIDEO_LIST",X["post.comments"]="LATENCY_ACTION_CREATOR_POST_COMMENTS",X["post.edit"]="LATENCY_ACTION_CREATOR_POST_EDIT",X.prebuffer="LATENCY_ACTION_PREBUFFER",X.prefetch="LATENCY_ACTION_PREFETCH",X.profile_settings="LATENCY_ACTION_KIDS_PROFILE_SETTINGS",
X.profile_switcher="LATENCY_ACTION_LOGIN",X.reel_watch="LATENCY_ACTION_REEL_WATCH",X.results="LATENCY_ACTION_RESULTS",X["promotion.edit"]="LATENCY_ACTION_CREATOR_PROMOTION_EDIT",X.red="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",X.premium="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",X.search_overview_answer="LATENCY_ACTION_SEARCH_OVERVIEW_ANSWER",X.search_ui="LATENCY_ACTION_SEARCH_UI",X.search_suggest="LATENCY_ACTION_SUGGEST",X.search_zero_state="LATENCY_ACTION_SEARCH_ZERO_STATE",X.secret_code="LATENCY_ACTION_KIDS_SECRET_CODE",
X.seek="LATENCY_ACTION_PLAYER_SEEK",X.settings="LATENCY_ACTION_SETTINGS",X.store="LATENCY_ACTION_STORE",X.supervision_dashboard="LATENCY_ACTION_KIDS_SUPERVISION_DASHBOARD",X.tenx="LATENCY_ACTION_TENX",X.video_to_ad="LATENCY_ACTION_VIDEO_TO_AD",X.watch="LATENCY_ACTION_WATCH",X.watch_it_again="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN",X["watch,watch7"]="LATENCY_ACTION_WATCH",X["watch,watch7_html5"]="LATENCY_ACTION_WATCH",X["watch,watch7ad"]="LATENCY_ACTION_WATCH",X["watch,watch7ad_html5"]="LATENCY_ACTION_WATCH",
X.wn_comments="LATENCY_ACTION_LOAD_COMMENTS",X.ww_rqs="LATENCY_ACTION_WHO_IS_WATCHING",X["video.analytics"]="LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS",X["video.claims"]="LATENCY_ACTION_CREATOR_VIDEO_CLAIMS",X["video.comments"]="LATENCY_ACTION_CREATOR_VIDEO_COMMENTS",X["video.copyright"]="LATENCY_ACTION_CREATOR_VIDEO_COPYRIGHT",X["video.edit"]="LATENCY_ACTION_CREATOR_VIDEO_EDIT",X["video.editor"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR",X["video.editor_async"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC",
X["video.live_settings"]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS",X["video.live_streaming"]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING",X["video.monetization"]="LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION",X["video.policy"]="LATENCY_ACTION_CREATOR_VIDEO_POLICY",X["video.rights_management"]="LATENCY_ACTION_CREATOR_VIDEO_RIGHTS_MANAGEMENT",X["video.translations"]="LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS",X.voice_assistant="LATENCY_ACTION_VOICE_ASSISTANT",X.cast_load_by_entity_to_watch="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH",
X.networkless_performance="LATENCY_ACTION_NETWORKLESS_PERFORMANCE",X.gel_compression="LATENCY_ACTION_GEL_COMPRESSION",X.gel_jspb_serialize="LATENCY_ACTION_GEL_JSPB_SERIALIZE",X);function Gv(a,b){Cp.call(this,1,arguments);this.timer=b}
y(Gv,Cp);var Hv=new Dp("aft-recorded",Gv);var Iv=C.ytLoggingLatencyUsageStats_||{};D("ytLoggingLatencyUsageStats_",Iv);function Jv(){this.h=0}
function Kv(){Jv.h||(Jv.h=new Jv);return Jv.h}
Jv.prototype.tick=function(a,b,c,d){Lv(this,"tick_"+a+"_"+b)||Hn("latencyActionTicked",{tickName:a,clientActionNonce:b},{timestamp:c,cttAuthInfo:d})};
Jv.prototype.info=function(a,b,c){var d=Object.keys(a).join("");Lv(this,"info_"+d+"_"+b)||(a=Object.assign({},a),a.clientActionNonce=b,Hn("latencyActionInfo",a,{cttAuthInfo:c}))};
Jv.prototype.jspbInfo=function(){};
Jv.prototype.span=function(a,b,c){var d=Object.keys(a).join("");Lv(this,"span_"+d+"_"+b)||(a.clientActionNonce=b,Hn("latencyActionSpan",a,{cttAuthInfo:c}))};
function Lv(a,b){Iv[b]=Iv[b]||{count:0};var c=Iv[b];c.count++;c.time=W();a.h||(a.h=Sm(function(){var d=W(),e;for(e in Iv)Iv[e]&&d-Iv[e].time>6E4&&delete Iv[e];a&&(a.h=0)},5E3));
return c.count>5?(c.count===6&&Math.random()*1E5<1&&(c=new V("CSI data exceeded logging limit with key",b.split("_")),b.indexOf("plev")>=0||qt(c)),!0):!1}
;var Mv=window;function Nv(){this.timing={};this.clearResourceTimings=function(){};
this.webkitClearResourceTimings=function(){};
this.mozClearResourceTimings=function(){};
this.msClearResourceTimings=function(){};
this.oClearResourceTimings=function(){}}
function Ov(){var a;if(U("csi_use_performance_navigation_timing")||U("csi_use_performance_navigation_timing_tvhtml5")){var b,c,d,e=Y==null?void 0:(a=Y.getEntriesByType)==null?void 0:(b=a.call(Y,"navigation"))==null?void 0:(c=b[0])==null?void 0:(d=c.toJSON)==null?void 0:d.call(c);e?(e.requestStart=Pv(e.requestStart),e.responseEnd=Pv(e.responseEnd),e.redirectStart=Pv(e.redirectStart),e.redirectEnd=Pv(e.redirectEnd),e.domainLookupEnd=Pv(e.domainLookupEnd),e.connectStart=Pv(e.connectStart),e.connectEnd=
Pv(e.connectEnd),e.responseStart=Pv(e.responseStart),e.secureConnectionStart=Pv(e.secureConnectionStart),e.domainLookupStart=Pv(e.domainLookupStart),e.isPerformanceNavigationTiming=!0,a=e):a=Y.timing}else a=U("csi_performance_timing_to_object")?JSON.parse(JSON.stringify(Y.timing)):Y.timing;return a}
function Pv(a){return Math.round(Qv()+a)}
function Qv(){return(U("csi_use_time_origin")||U("csi_use_time_origin_tvhtml5"))&&Y.timeOrigin?Math.floor(Y.timeOrigin):Y.timing.navigationStart}
var Y=Mv.performance||Mv.mozPerformance||Mv.msPerformance||Mv.webkitPerformance||new Nv;var Rv=!1,Sv=!1,Tv={'script[name="scheduler/scheduler"]':"sj",'script[name="player/base"]':"pj",'link[rel="preload"][name="player/embed"]':"pej",'link[rel="stylesheet"][name="www-player"]':"pc",'link[rel="stylesheet"][name="player/www-player"]':"pc",'script[name="desktop_polymer/desktop_polymer"]':"dpj",'link[rel="import"][name="desktop_polymer"]':"dph",'script[name="mobile-c3"]':"mcj",'link[rel="stylesheet"][name="mobile-c3"]':"mcc",'script[name="player-plasma-ias-phone/base"]':"mcppj",'script[name="player-plasma-ias-tablet/base"]':"mcptj",
'link[rel="stylesheet"][name="mobile-polymer-player-ias"]':"mcpc",'link[rel="stylesheet"][name="mobile-polymer-player-svg-ias"]':"mcpsc",'script[name="mobile_blazer_core_mod"]':"mbcj",'link[rel="stylesheet"][name="mobile_blazer_css"]':"mbc",'script[name="mobile_blazer_logged_in_users_mod"]':"mbliuj",'script[name="mobile_blazer_logged_out_users_mod"]':"mblouj",'script[name="mobile_blazer_noncore_mod"]':"mbnj","#player_css":"mbpc",'script[name="mobile_blazer_desktopplayer_mod"]':"mbpj",'link[rel="stylesheet"][name="mobile_blazer_tablet_css"]':"mbtc",
'script[name="mobile_blazer_watch_mod"]':"mbwj"};Xa(Y.clearResourceTimings||Y.webkitClearResourceTimings||Y.mozClearResourceTimings||Y.msClearResourceTimings||Y.oClearResourceTimings||Bd,Y);function Uv(a,b){if(!U("web_csi_action_sampling_enabled")||!wv(b).actionDisabled){var c=Ev(b||"");zu(c.info,a);a.loadType&&(c=a.loadType,yv(b).loadType=c);zu(Bv(b),a);c=Cv(b);b=wv(b).cttAuthInfo;Kv().info(a,c,b)}}
function Vv(){var a,b,c,d;return((d=$r().resolve(new Ur(rp))==null?void 0:(a=sp())==null?void 0:(b=a.loggingHotConfig)==null?void 0:(c=b.csiConfig)==null?void 0:c.debugTicks)!=null?d:[]).map(function(e){return Object.values(e)[0]})}
function Z(a,b,c){if(!U("web_csi_action_sampling_enabled")||!wv(c).actionDisabled){var d=Cv(c),e;if(e=U("web_csi_debug_sample_enabled")&&d){($r().resolve(new Ur(rp))==null?0:sp())&&!Sv&&(Sv=!0,Z("gcfl",W(),c));var f,g,h;e=($r().resolve(new Ur(rp))==null?void 0:(f=sp())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.csiConfig)==null?void 0:h.debugSampleWeight)||0;if(f=e!==0)b:{f=Vv();if(f.length>0)for(g=0;g<f.length;g++)if(a===f[g]){f=!0;break b}f=!1}if(f){for(g=f=0;g<d.length;g++)f=f*31+d.charCodeAt(g),
g<d.length-1&&(f%=Math.pow(2,47));e=f%1E5%e!==0;wv(c).debugTicksExcludedLogged||(f={},f.debugTicksExcluded=e,Uv(f,c));wv(c).debugTicksExcludedLogged=!0}else e=!1}if(!e){if(a[0]!=="_"&&(e=a,f=b,Y.mark))if(e.startsWith("mark_")||(e="mark_"+e),c&&(e+=" ("+c+")"),f===void 0||U("web_csi_disable_alt_time_performance_mark"))Y.mark(e);else{f-=Y.timeOrigin||Y.timing.navigationStart;try{Y.mark(e,{startTime:f})}catch(k){}}e=Ev(c||"");e.tick[a]=b||W();if(e.callback&&e.callback[a])for(e=x(e.callback[a]),f=e.next();!f.done;f=
e.next())f=f.value,f();e=Av(c);e.gelTicks&&(e.gelTicks[a]=!0);f=zv(c);e=b||W();U("log_repeated_ytcsi_ticks")?a in f||(f[a]=e):f[a]=e;f=wv(c).cttAuthInfo;a==="_start"?(a=Kv(),Lv(a,"baseline_"+d)||Hn("latencyActionBaselined",{clientActionNonce:d},{timestamp:b,cttAuthInfo:f})):Kv().tick(a,d,b,f);Wv(c);return e}}}
function Xv(){var a=document;if("visibilityState"in a)a=a.visibilityState;else{var b=qr+"VisibilityState";a=b in a?a[b]:void 0}switch(a){case "hidden":return 0;case "visible":return 1;case "prerender":return 2;case "unloaded":return 3;default:return-1}}
function Yv(){function a(f,g,h){g=g.match("_rid")?g.split("_rid")[0]:g;typeof h==="number"&&(h=JSON.stringify(h));f.requestIds?f.requestIds.push({endpoint:g,id:h}):f.requestIds=[{endpoint:g,id:h}]}
for(var b={},c=x(Object.entries(T("TIMING_INFO",{}))),d=c.next();!d.done;d=c.next()){var e=x(d.value);d=e.next().value;e=e.next().value;switch(d){case "GetBrowse_rid":a(b,d,e);break;case "GetGuide_rid":a(b,d,e);break;case "GetHome_rid":a(b,d,e);break;case "GetPlayer_rid":a(b,d,e);break;case "GetSearch_rid":a(b,d,e);break;case "GetSettings_rid":a(b,d,e);break;case "GetTrending_rid":a(b,d,e);break;case "GetWatchNext_rid":a(b,d,e);break;case "yt_red":b.isRedSubscriber=!!e;break;case "yt_ad":b.isMonetized=
!!e}}return b}
function Zv(a,b){a=document.querySelector(a);if(!a)return!1;var c="",d=a.nodeName;d==="SCRIPT"?(c=a.src,c||(c=a.getAttribute("data-timing-href"))&&(c=window.location.protocol+c)):d==="LINK"&&(c=a.href);$b(window)&&a.setAttribute("nonce",$b(window));return c?(a=Y.getEntriesByName(c))&&a[0]&&(a=a[0],c=Qv(),Z("rsf_"+b,c+Math.round(a.fetchStart)),Z("rse_"+b,c+Math.round(a.responseEnd)),a.transferSize!==void 0&&a.transferSize===0)?!0:!1:!1}
function $v(){var a=window.location.protocol,b=Y.getEntriesByType("resource");b=Eb(b,function(c){return c.name.indexOf(a+"//fonts.gstatic.com/s/")===0});
(b=Gb(b,function(c,d){return d.duration>c.duration?d:c},{duration:0}))&&b.startTime>0&&b.responseEnd>0&&(Z("wffs",Pv(b.startTime)),Z("wffe",Pv(b.responseEnd)))}
function aw(a){var b=bw("aft",a);if(b)return b;b=T((a||"")+"TIMING_AFT_KEYS",["ol"]);for(var c=b.length,d=0;d<c;d++){var e=bw(b[d],a);if(e)return e}return NaN}
function bw(a,b){if(a=zv(b)[a])return typeof a==="number"?a:a[a.length-1]}
function Wv(a){var b=bw("_start",a),c=aw(a),d=U("enable_cow_info_csi")||!Rv;b&&c&&d&&(Ip(Hv,new Gv(Math.round(c-b),a)),Rv=!0)}
function cw(){if(Y.getEntriesByType){var a=Y.getEntriesByType("paint");if(a=Hb(a,function(b){return b.name==="first-paint"}))return Pv(a.startTime)}a=Y.timing;
return a.xe?Math.max(0,a.xe):0}
;function dw(a,b){tl(function(){Ev("").info.actionType=a;b&&pl("TIMING_AFT_KEYS",b);pl("TIMING_ACTION",a);var c=Yv();Object.keys(c).length>0&&Uv(c);c={isNavigation:!0,actionType:Fv[T("TIMING_ACTION")]||"LATENCY_ACTION_UNKNOWN"};var d=T("PREVIOUS_ACTION");d&&(c.previousAction=Fv[d]||"LATENCY_ACTION_UNKNOWN");if(d=T("CLIENT_PROTOCOL"))c.httpProtocol=d;if(d=T("CLIENT_TRANSPORT"))c.transportProtocol=d;(d=It())&&d!=="UNDEFINED_CSN"&&(c.clientScreenNonce=d);d=Xv();if(d===1||d===-1)c.isVisible=!0;yv();xv();
c.loadType="cold";d=xv();var e=Ov(),f=Qv(),g=T("CSI_START_TIMESTAMP_MILLIS",0);g>0&&!U("embeds_web_enable_csi_start_override_killswitch")&&(f=g);f&&(Z("srt",e.responseStart),d.prerender!==1&&Z("_start",f,void 0));d=cw();d>0&&Z("fpt",d);d=Ov();d.isPerformanceNavigationTiming&&Uv({performanceNavigationTiming:!0},void 0);Z("nreqs",d.requestStart,void 0);Z("nress",d.responseStart,void 0);Z("nrese",d.responseEnd,void 0);d.redirectEnd-d.redirectStart>0&&(Z("nrs",d.redirectStart,void 0),Z("nre",d.redirectEnd,
void 0));d.domainLookupEnd-d.domainLookupStart>0&&(Z("ndnss",d.domainLookupStart,void 0),Z("ndnse",d.domainLookupEnd,void 0));d.connectEnd-d.connectStart>0&&(Z("ntcps",d.connectStart,void 0),Z("ntcpe",d.connectEnd,void 0));d.secureConnectionStart>=Qv()&&d.connectEnd-d.secureConnectionStart>0&&(Z("nstcps",d.secureConnectionStart,void 0),Z("ntcpe",d.connectEnd,void 0));Y&&"getEntriesByType"in Y&&$v();d=[];if(document.querySelector&&Y&&Y.getEntriesByName)for(var h in Tv)Tv.hasOwnProperty(h)&&(e=Tv[h],
Zv(h,e)&&d.push(e));if(d.length>0)for(c.resourceInfo=[],h=x(d),d=h.next();!d.done;d=h.next())c.resourceInfo.push({resourceCache:d.value});Uv(c);c=Av();c.preLoggedGelInfos||(c.preLoggedGelInfos=[]);h=c.preLoggedGelInfos;c=Bv();d=void 0;for(e=0;e<h.length;e++)if(f=h[e],f.loadType){d=f.loadType;break}if(yv().loadType==="cold"&&(c.loadType==="cold"||d==="cold")){d=zv();e=Av();e=e.gelTicks?e.gelTicks:e.gelTicks={};for(var k in d)if(!(k in e))if(typeof d[k]==="number")Z(k,bw(k));else if(U("log_repeated_ytcsi_ticks"))for(f=
x(d[k]),g=f.next();!g.done;g=f.next())g=g.value,Z(k.slice(1),g);k={};d=!1;h=x(h);for(e=h.next();!e.done;e=h.next())d=e.value,zu(c,d),zu(k,d),d=!0;d&&Uv(k)}D("ytglobal.timingready_",!0);k=T("TIMING_ACTION");E("ytglobal.timingready_")&&k&&ew()&&aw()&&Wv()})()}
function ew(){return tl(function(){return fw()})()}
function gw(a,b,c){tl(Uv)(a,b,c===void 0?!1:c)}
function hw(a,b,c){return tl(Z)(a,b,c)}
function fw(){return tl(function(){return"_start"in zv()})()}
function iw(){tl(function(){var a=Cv();requestAnimationFrame(function(){setTimeout(function(){a===Cv()&&hw("ol",void 0,void 0)},0)})})()}
var jw=window;jw.ytcsi&&(jw.ytcsi.infoGel=gw,jw.ytcsi.tick=hw);var kw="tokens consistency mss client_location entities adblock_detection response_received_commands store PLAYER_PRELOAD".split(" "),lw=["type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.BrowseResponse","type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.PlayerResponse"];function mw(a,b,c,d){this.v=a;this.fa=b;this.l=c;this.j=d;this.i=void 0;this.h=new Map;a.Rb||(a.Rb={});a.Rb=Object.assign({},vv,a.Rb)}
function nw(a,b,c,d){if(mw.h!==void 0){if(d=mw.h,a=[a!==d.v,b!==d.fa,c!==d.l,!1,!1,!1,void 0!==d.i],a.some(function(e){return e}))throw new V("InnerTubeTransportService is already initialized",a);
}else mw.h=new mw(a,b,c,d)}
function ow(a){var b={signalServiceEndpoint:{signal:"GET_DATASYNC_IDS"}};var c=c===void 0?xm:c;var d=pw(a,b);return d?new Ud(function(e,f){var g,h,k,l,m;return A(function(p){switch(p.h){case 1:return p.yield(d,2);case 2:g=p.i;h=g.v(b,void 0,c);if(!h){f(new V("Error: Failed to build request for command.",b));p.B(0);break}Zu(h.input);l=((k=h.ib)==null?void 0:k.mode)==="cors"?"cors":void 0;if(a.l.df){var t=h.config,r;t=t==null?void 0:(r=t.Wb)==null?void 0:r.sessionIndex;r=wm(0,{sessionIndex:t});m=Object.assign({},
kv(l),r);p.B(4);break}return p.yield(qw(h.config,l),5);case 5:m=p.i;case 4:e(rw(a,h,m)),p.h=0}})}):Zd(new V("Error: No request builder found for command.",b))}
function sw(a,b,c){var d;if(b&&!(b==null?0:(d=b.sequenceMetaData)==null?0:d.skipProcessing)&&a.j){d=x(kw);for(var e=d.next();!e.done;e=d.next())e=e.value,a.j[e]&&a.j[e].handleResponse(b,c)}}
function rw(a,b,c){var d=d===void 0?function(){}:d;
var e,f,g,h,k,l,m,p,t,r,v,u,z,F,J,O,S,da,va,P,ea,na,La,Ka,Qg,Rg,Ar,Br,Cr;return A(function(ha){switch(ha.h){case 1:ha.B(2);break;case 3:if((e=ha.i)&&!e.isExpired())return ha.return(Promise.resolve(e.h()));case 2:if(!((f=b)==null?0:(g=f.Oa)==null?0:g.context)){ha.B(4);break}h=b.Oa.context;ha.B(5);break;case 5:k=x([]),l=k.next();case 8:if(l.done){ha.B(4);break}m=l.value;return ha.yield(m.Sg(h),9);case 9:l=k.next();ha.B(8);break;case 4:if((p=a.i)==null||!p.Xg(b.input,b.Oa)){ha.B(12);break}return ha.yield(a.i.Og(b.input,
b.Oa),13);case 13:return t=ha.i,U("kevlar_process_local_innertube_responses_killswitch")||sw(a,t,b),ha.return(t);case 12:return(u=(v=b.config)==null?void 0:v.Vg)&&a.h.has(u)?r=a.h.get(u):(z=JSON.stringify(b.Oa),O=(J=(F=b.ib)==null?void 0:F.headers)!=null?J:{},b.ib=Object.assign({},b.ib,{headers:Object.assign({},O,c)}),S=Object.assign({},b.ib),b.ib.method==="POST"&&(S=Object.assign({},S,{body:z})),((da=b.config)==null?0:da.He)&&hw(b.config.He),va=function(){return a.fa.fetch(b.input,S,b.config)},r=
va(),u&&a.h.set(u,r)),ha.yield(r,14);
case 14:if((P=ha.i)&&"error"in P&&((ea=P)==null?0:(na=ea.error)==null?0:na.details))for(La=P.error.details,Ka=x(La),Qg=Ka.next();!Qg.done;Qg=Ka.next())Rg=Qg.value,(Ar=Rg["@type"])&&lw.indexOf(Ar)>-1&&(delete Rg["@type"],P=Rg);u&&a.h.has(u)&&a.h.delete(u);((Br=b.config)==null?0:Br.Ie)&&hw(b.config.Ie);if(P||(Cr=a.i)==null||!Cr.Gg(b.input,b.Oa)){ha.B(15);break}return ha.yield(a.i.Ng(b.input,b.Oa),16);case 16:P=ha.i;case 15:return sw(a,P,b),d(),ha.return(P||void 0)}})}
function pw(a,b){a:{a=a.v;var c,d=(c=ys(b,Zk))==null?void 0:c.signal;if(d&&a.Rb&&(c=a.Rb[d])){var e=c();break a}var f;if((c=(f=ys(b,Xk))==null?void 0:f.request)&&a.Rd&&(f=a.Rd[c])){e=f();break a}for(e in b)if(a.Xc[e]&&(b=a.Xc[e])){e=b();break a}e=void 0}if(e!==void 0)return Promise.resolve(e)}
function qw(a,b){var c,d,e,f;return A(function(g){if(g.h==1){e=(c=a)==null?void 0:(d=c.Wb)==null?void 0:d.sessionIndex;var h=g.yield;var k=wm(0,{sessionIndex:e});if(!(k instanceof Ud)){var l=new Ud(Bd);Vd(l,2,k);k=l}return h.call(g,k,2)}f=g.i;return g.return(Promise.resolve(Object.assign({},kv(b),f)))})}
;var tw=new Tr("INNERTUBE_TRANSPORT_TOKEN");function uw(){}
y(uw,sv);uw.prototype.j=function(){return lu};
uw.prototype.i=function(a){return ys(a,gl)||void 0};
uw.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params);c.botguardResponse&&(a.botguardResponse=c.botguardResponse);c.feature&&(a.clientFeature=c.feature)};
fa.Object.defineProperties(uw.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function vw(){}
y(vw,sv);vw.prototype.j=function(){return mu};
vw.prototype.i=function(a){return ys(a,fl)||void 0};
vw.prototype.h=function(a,b){b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params)};
fa.Object.defineProperties(vw.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function ww(){}
y(ww,sv);ww.prototype.j=function(){return iu};
ww.prototype.i=function(a){return ys(a,al)||void 0};
ww.prototype.h=function(a,b,c){a.feedbackTokens=[];b.feedbackToken&&a.feedbackTokens.push(b.feedbackToken);if(b=b.cpn||c.cpn)a.feedbackContext={cpn:b};a.isFeedbackTokenUnencrypted=!!c.is_feedback_token_unencrypted;a.shouldMerge=!1;c.extra_feedback_tokens&&(a.shouldMerge=!0,a.feedbackTokens=a.feedbackTokens.concat(c.extra_feedback_tokens))};
fa.Object.defineProperties(ww.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function xw(){}
y(xw,sv);xw.prototype.j=function(){return ju};
xw.prototype.i=function(a){return ys(a,el)||void 0};
xw.prototype.h=function(a,b){b.params&&(a.params=b.params);b.secondaryParams&&(a.secondaryParams=b.secondaryParams)};function yw(){}
y(yw,sv);yw.prototype.j=function(){return ku};
yw.prototype.i=function(a){return ys(a,dl)||void 0};
yw.prototype.h=function(a,b){b.actions&&(a.actions=b.actions);b.params&&(a.params=b.params);b.playlistId&&(a.playlistId=b.playlistId)};function zw(){}
y(zw,sv);zw.prototype.j=function(){return hu};
zw.prototype.i=function(a){return ys(a,cl)};
zw.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);c.includeListId&&(a.includeListId=!0)};function Aw(a,b){var c=B.apply(2,arguments);a=a===void 0?0:a;V.call(this,b,c);this.errorType=a;Object.setPrototypeOf(this,this.constructor.prototype)}
y(Aw,V);var Bw=new Tr("NETWORK_SLI_TOKEN");function Cw(a){this.h=a}
Cw.prototype.fetch=function(a,b,c){var d=this,e;return A(function(f){e=Dw(d,a,b);return f.return(fetch(e).then(function(g){return d.handleResponse(g,c)}).catch(function(g){qt(g);
if((c==null?0:c.Xd)&&g instanceof Aw&&g.errorType===1)return Promise.reject(g)}))})};
function Dw(a,b,c){if(a.h){var d=mc(nc(5,xc(b,"key")))||"/UNKNOWN_PATH";a.h.start(d)}a=c;U("wug_networking_gzip_request")&&(a=jq(c));return new window.Request(b,a)}
Cw.prototype.handleResponse=function(a,b){var c=a.text().then(function(d){if((b==null?0:b.qe)&&a.ok)return dh(b.qe,d);d=d.replace(")]}'","");if((b==null?0:b.Xd)&&d)try{var e=JSON.parse(d)}catch(g){throw new Aw(1,"JSON parsing failed after fetch");}var f;return(f=e)!=null?f:JSON.parse(d)});
a.redirected||a.ok?this.h&&this.h.success():(this.h&&this.h.Jg(),c=c.then(function(d){qt(new V("Error: API fetch failed",a.status,a.url,d));return Object.assign({},d,{errorMetadata:{status:a.status}})}));
return c};
Cw[Sr]=[new Ur(Bw)];var Ew=new Tr("NETWORK_MANAGER_TOKEN");var Fw;function Gw(){var a,b,c;return A(function(d){if(d.h==1)return a=$r().resolve(tw),a?d.yield(ow(a),2):(qt(Error("InnertubeTransportService unavailable in fetchDatasyncIds")),d.return(void 0));if(b=d.i){if(b.errorMetadata)return qt(Error("Datasync IDs fetch responded with "+b.errorMetadata.status+": "+b.error)),d.return(void 0);c=b.Hg;return d.return(c)}qt(Error("Network request to get Datasync IDs failed."));return d.return(void 0)})}
;function Hw(){var a;return(a=T("WEB_PLAYER_CONTEXT_CONFIGS"))==null?void 0:a.WEB_PLAYER_CONTEXT_CONFIG_ID_EMBEDDED_PLAYER}
;var Iw=C.caches,Jw;function Kw(a){var b=a.indexOf(":");return b===-1?{nd:a}:{nd:a.substring(0,b),datasyncId:a.substring(b+1)}}
function Lw(){return A(function(a){if(Jw!==void 0)return a.return(Jw);Jw=new Promise(function(b){var c;return A(function(d){switch(d.h){case 1:return Aa(d,2),d.yield(Iw.open("test-only"),4);case 4:return d.yield(Iw.delete("test-only"),5);case 5:d.h=3;d.l=0;break;case 2:if(c=Ba(d),c instanceof Error&&c.name==="SecurityError")return b(!1),d.return();case 3:b("caches"in window),d.h=0}})});
return a.return(Jw)})}
function Mw(a){var b,c,d,e,f,g,h;A(function(k){if(k.h==1)return k.yield(Lw(),2);if(k.h!=3){if(!k.i)return k.return(!1);b=[];return k.yield(Iw.keys(),3)}c=k.i;d=x(c);for(e=d.next();!e.done;e=d.next())f=e.value,g=Kw(f),h=g.datasyncId,!h||a.includes(h)||b.push(Iw.delete(f));return k.return(Promise.all(b).then(function(l){return l.some(function(m){return m})}))})}
function Nw(){var a,b,c,d,e,f,g;return A(function(h){if(h.h==1)return h.yield(Lw(),2);if(h.h!=3){if(!h.i)return h.return(!1);a=Qm("cache contains other");return h.yield(Iw.keys(),3)}b=h.i;c=x(b);for(d=c.next();!d.done;d=c.next())if(e=d.value,f=Kw(e),(g=f.datasyncId)&&g!==a)return h.return(!0);return h.return(!1)})}
;function Ow(){try{return!!self.sessionStorage}catch(a){return!1}}
;function Pw(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function Qw(a){if(Ow()){var b=Object.keys(window.sessionStorage);b=x(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=Pw(c);d===void 0||a.includes(d)||self.sessionStorage.removeItem(c)}}}
function Rw(){if(!Ow())return!1;var a=Qm(),b=Object.keys(window.sessionStorage);b=x(b);for(var c=b.next();!c.done;c=b.next())if(c=Pw(c.value),c!==void 0&&c!==a)return!0;return!1}
;function Sw(){Gw().then(function(a){a&&(Xo(a),Mw(a),Wu(a),Qw(a))})}
function Tw(){var a=new br;cj.pa(function(){var b,c,d,e,f;return A(function(g){switch(g.h){case 1:if(U("ytidb_clear_optimizations_killswitch")){g.B(2);break}b=Qm("clear");if(b.startsWith("V")&&b.endsWith("||")){var h=[b];Xo(h);Mw(h);Wu(h);Qw(h);return g.return()}c=Xu();d=Rw();return g.yield(Nw(),3);case 3:return e=g.i,g.yield(Yo(),4);case 4:if(f=g.i,!(c||d||e||f))return g.return();case 2:a.va()?Sw():a.h.add("publicytnetworkstatus-online",Sw,!0,void 0,void 0),g.h=0}})})}
;function Uw(){this.state=1;this.h=null}
n=Uw.prototype;n.initialize=function(a,b,c){if(a.program){var d,e=(d=a.interpreterUrl)!=null?d:null;if(a.interpreterSafeScript){var f=a.interpreterSafeScript;f?((f=f.privateDoNotAccessOrElseSafeScriptWrappedValue)?(d=fb(),f=new ac(d?d.createScript(f):f)):f=null,d=f):d=null}else d=(f=a.interpreterScript)!=null?f:null;a.interpreterSafeUrl&&(e=Rk(a.interpreterSafeUrl).toString());Vw(this,d,e,a.program,b,c)}else qt(Error("Cannot initialize botguard without program"))};
function Vw(a,b,c,d,e,f){var g=g===void 0?"trayride":g;c?(a.state=2,pu(c,function(){window[g]?Ww(a,d,g,e):(a.state=3,ru(c),qt(new V("Unable to load Botguard","from "+c)))},f)):b?(f=Gd("SCRIPT"),b instanceof ac?cc(f,b):f.textContent=b,f.nonce=$b(window),document.head.appendChild(f),document.head.removeChild(f),window[g]?Ww(a,d,g,e):(a.state=4,qt(new V("Unable to load Botguard from JS")))):qt(new V("Unable to load VM; no url or JS provided"))}
n.isLoading=function(){return this.state===2};
function Ww(a,b,c,d){a.state=5;try{var e=new Ni({program:b,ee:c,Fe:U("att_web_record_metrics"),Ea:"aGIf"});e.Ye.then(function(){a.state=6;d&&d(b)});
a.Mc(e)}catch(f){a.state=7,f instanceof Error&&qt(f)}}
n.invoke=function(a){a=a===void 0?{}:a;return this.Qc()?this.Ed({Yc:a}):null};
n.dispose=function(){this.Mc(null);this.state=8};
n.Qc=function(){return!!this.h};
n.Ed=function(a){return this.h.yd(a)};
n.Mc=function(a){Cc(this.h);this.h=a};var Xw=[],Yw=!1;function Zw(){if(!U("disable_biscotti_fetch_for_ad_blocker_detection")&&!U("disable_biscotti_fetch_entirely_for_all_web_clients")&&Tt()){var a=T("PLAYER_VARS",{});if(Pb(a)!="1"&&!Ut(a)){var b=function(){Yw=!0;"google_ad_status"in window?pl("DCLKSTAT",1):pl("DCLKSTAT",2)};
try{pu("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Xw.push(cj.pa(function(){if(!(Yw||"google_ad_status"in window)){try{tu("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Yw=!0;pl("DCLKSTAT",3)}},5E3))}}}
function $w(){var a=Number(T("DCLKSTAT",0));return isNaN(a)?0:a}
;function ax(){var a=E("yt.abuse.playerAttLoader");return a&&["bgvma","bgvmb","bgvmc"].every(function(b){return b in a})?a:null}
;function bx(){Uw.apply(this,arguments)}
y(bx,Uw);bx.prototype.Mc=function(a){var b;(b=ax())==null||b.bgvma();a?(b={bgvma:a.dispose.bind(a),bgvmb:a.snapshot.bind(a),bgvmc:a.yd.bind(a)},D("yt.abuse.playerAttLoader",b),D("yt.abuse.playerAttLoaderRun",function(c){return a.snapshot(c)})):(D("yt.abuse.playerAttLoader",null),D("yt.abuse.playerAttLoaderRun",null))};
bx.prototype.Qc=function(){return!!ax()};
bx.prototype.Ed=function(a){return ax().bgvmc(a)};function cx(a){js.call(this,a===void 0?"document_active":a);var b=this;this.l=10;this.h=new Map;this.transitions=[{from:"document_active",to:"document_disposed_preventable",action:this.H},{from:"document_active",to:"document_disposed",action:this.v},{from:"document_disposed_preventable",to:"document_disposed",action:this.v},{from:"document_disposed_preventable",to:"flush_logs",action:this.m},{from:"document_disposed_preventable",to:"document_active",action:this.i},{from:"document_disposed",to:"flush_logs",
action:this.m},{from:"document_disposed",to:"document_active",action:this.i},{from:"document_disposed",to:"document_disposed",action:function(){}},
{from:"flush_logs",to:"document_active",action:this.i}];window.addEventListener("pagehide",function(c){b.transition("document_disposed",{event:c})});
window.addEventListener("beforeunload",function(c){b.transition("document_disposed_preventable",{event:c})})}
y(cx,js);cx.prototype.H=function(a,b){if(!this.h.get("document_disposed_preventable")){a(b==null?void 0:b.event);var c,d;if((b==null?0:(c=b.event)==null?0:c.defaultPrevented)||(b==null?0:(d=b.event)==null?0:d.returnValue)){b.event.returnValue||(b.event.returnValue=!0);b.event.defaultPrevented||b.event.preventDefault();this.h=new Map;this.transition("document_active");return}}this.h.set("document_disposed_preventable",!0);this.h.get("document_disposed")?this.transition("flush_logs"):this.transition("document_disposed")};
cx.prototype.v=function(a,b){this.h.get("document_disposed")?this.transition("document_active"):(a(b==null?void 0:b.event),this.h.set("document_disposed",!0),this.transition("flush_logs"))};
cx.prototype.m=function(a,b){a(b==null?void 0:b.event);this.transition("document_active")};
cx.prototype.i=function(){this.h=new Map};function dx(a){js.call(this,a===void 0?"document_visibility_unknown":a);var b=this;this.transitions=[{from:"document_visibility_unknown",to:"document_visible",action:this.i},{from:"document_visibility_unknown",to:"document_hidden",action:this.h},{from:"document_visibility_unknown",to:"document_foregrounded",action:this.m},{from:"document_visibility_unknown",to:"document_backgrounded",action:this.v},{from:"document_visible",to:"document_hidden",action:this.h},{from:"document_visible",to:"document_foregrounded",
action:this.m},{from:"document_visible",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_hidden",action:this.h},{from:"document_foregrounded",to:"document_foregrounded",action:this.m},{from:"document_hidden",to:"document_visible",action:this.i},{from:"document_hidden",to:"document_backgrounded",action:this.v},{from:"document_hidden",to:"document_hidden",action:this.h},{from:"document_backgrounded",to:"document_hidden",
action:this.h},{from:"document_backgrounded",to:"document_backgrounded",action:this.v},{from:"document_backgrounded",to:"document_visible",action:this.i}];document.addEventListener("visibilitychange",function(c){document.visibilityState==="visible"?b.transition("document_visible",{event:c}):b.transition("document_hidden",{event:c})});
U("visibility_lifecycles_dynamic_backgrounding")&&(window.addEventListener("blur",function(c){b.transition("document_backgrounded",{event:c})}),window.addEventListener("focus",function(c){b.transition("document_foregrounded",{event:c})}))}
y(dx,js);dx.prototype.i=function(a,b){a(b==null?void 0:b.event);U("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_foregrounded")};
dx.prototype.h=function(a,b){a(b==null?void 0:b.event);U("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_backgrounded")};
dx.prototype.v=function(a,b){a(b==null?void 0:b.event)};
dx.prototype.m=function(a,b){a(b==null?void 0:b.event)};function ex(){this.l=new cx;this.v=new dx}
ex.prototype.install=function(){var a=B.apply(0,arguments),b=this;a.forEach(function(c){b.l.install(c)});
a.forEach(function(c){b.v.install(c)})};function fx(){this.l=[];this.i=new Map;this.h=new Map;this.j=new Set}
fx.prototype.clickCommand=function(a,b,c){var d=a.clickTrackingParams;c=c===void 0?0:c;if(d)if(c=It(c===void 0?0:c)){a=this.client;d=new Bt({trackingParams:d});var e=void 0;if(U("no_client_ve_attach_unless_shown")){var f=Ru(d,c);Nu.set(f,!0);Su(d,c)}e=e||"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK";f=Qu({cttAuthInfo:Kt(c)||void 0},c);d={csn:c,ve:d.getAsJson(),gestureType:e};b&&(d.clientData=b);c==="UNDEFINED_CSN"?Tu("visualElementGestured",f,d):a?ht("visualElementGestured",d,a,f):Hn("visualElementGestured",
d,f);b=!0}else b=!1;else b=!1;return b};
fx.prototype.stateChanged=function(a,b,c){this.visualElementStateChanged(new Bt({trackingParams:a}),b,c===void 0?0:c)};
fx.prototype.visualElementStateChanged=function(a,b,c){c=c===void 0?0:c;if(c===0&&this.j.has(c))this.l.push([a,b]);else{var d=c;d=d===void 0?0:d;c=It(d);a||(a=(a=Ft(d===void 0?0:d))?new Bt({veType:a,youtubeData:void 0,jspbYoutubeData:void 0}):null);var e=a;c&&e&&(a=this.client,d=Qu({cttAuthInfo:Kt(c)||void 0},c),b={csn:c,ve:e.getAsJson(),clientData:b},c==="UNDEFINED_CSN"?Tu("visualElementStateChanged",d,b):a?ht("visualElementStateChanged",b,a,d):Hn("visualElementStateChanged",b,d))}};
function gx(a,b){if(b===void 0)for(var c=Ht(),d=0;d<c.length;d++)c[d]!==void 0&&gx(a,c[d]);else a.i.forEach(function(e,f){(f=a.h.get(f))&&Pu(a.client,b,f,e)}),a.i.clear(),a.h.clear()}
;function hx(){ex.call(this);var a={};this.install((a.document_disposed={callback:this.h},a));U("combine_ve_grafts")&&(a={},this.install((a.document_disposed={callback:this.i},a)));a={};this.install((a.flush_logs={callback:this.j},a));U("web_log_cfg_cee_ks")||Sm(ix)}
y(hx,ex);hx.prototype.j=function(){Hn("finalPayload",{csn:It()})};
hx.prototype.h=function(){ut(vt)};
hx.prototype.i=function(){var a=gx;fx.h||(fx.h=new fx);a(fx.h)};
function ix(){var a=T("CLIENT_EXPERIMENT_EVENTS");if(a){var b=Xi();a=x(a);for(var c=a.next();!c.done;c=a.next())c=c.value,b(c)&&Hn("genericClientExperimentEvent",{eventType:c});delete ol.CLIENT_EXPERIMENT_EVENTS}}
;function jx(){}
function kx(){var a=E("ytglobal.storage_");a||(a=new jx,D("ytglobal.storage_",a));return a}
jx.prototype.estimate=function(){var a,b,c;return A(function(d){a=navigator;return((b=a.storage)==null?0:b.estimate)?d.return(a.storage.estimate()):((c=a.webkitTemporaryStorage)==null?0:c.queryUsageAndQuota)?d.return(lx()):d.return()})};
function lx(){var a=navigator;return new Promise(function(b,c){var d;(d=a.webkitTemporaryStorage)!=null&&d.queryUsageAndQuota?a.webkitTemporaryStorage.queryUsageAndQuota(function(e,f){b({usage:e,quota:f})},function(e){c(e)}):c(Error("webkitTemporaryStorage is not supported."))})}
D("ytglobal.storageClass_",jx);function Fn(a,b){var c=this;this.handleError=a;this.h=b;this.i=!1;self.document===void 0||self.addEventListener("beforeunload",function(){c.i=!0});
this.j=Math.random()<=.2}
Fn.prototype.Ha=function(a){this.handleError(a)};
Fn.prototype.logEvent=function(a,b){switch(a){case "IDB_DATA_CORRUPTED":U("idb_data_corrupted_killswitch")||this.h("idbDataCorrupted",b);break;case "IDB_UNEXPECTEDLY_CLOSED":this.h("idbUnexpectedlyClosed",b);break;case "IS_SUPPORTED_COMPLETED":U("idb_is_supported_completed_killswitch")||this.h("idbIsSupportedCompleted",b);break;case "QUOTA_EXCEEDED":mx(this,b);break;case "TRANSACTION_ENDED":this.j&&Math.random()<=.1&&this.h("idbTransactionEnded",b);break;case "TRANSACTION_UNEXPECTEDLY_ABORTED":a=
Object.assign({},b,{hasWindowUnloaded:this.i}),this.h("idbTransactionAborted",a)}};
function mx(a,b){kx().estimate().then(function(c){c=Object.assign({},b,{isSw:self.document===void 0,isIframe:self!==self.top,deviceStorageUsageMbytes:nx(c==null?void 0:c.usage),deviceStorageQuotaMbytes:nx(c==null?void 0:c.quota)});a.h("idbQuotaExceeded",c)})}
function nx(a){return typeof a==="undefined"?"-1":String(Math.ceil(a/1048576))}
;function ox(a,b,c){G.call(this);var d=this;this.channel="widget";this.sessionId=this.h=this.commands=this.l=null;this.targetOrigin="*";this.j=c||T("POST_MESSAGE_ORIGIN")||document.location.protocol+"//"+document.location.hostname;this.i=b||null;this.m=!!a;this.listener=function(e){a:if(!(d.j!=="*"&&e.origin!==d.j||d.i&&e.source!==d.i||typeof e.data!=="string")){try{var f=JSON.parse(e.data)}catch(g){break a}if(!(f==null||d.m&&(d.sessionId&&d.sessionId!==f.id||d.channel&&d.channel!==f.channel))&&f)switch(f.event){case "listening":e.origin!==
"null"&&(d.j=d.targetOrigin=e.origin);d.i=e.source;d.sessionId=f.id;d.h&&(d.h(),d.h=null);break;case "command":d.l&&(!d.commands||Cb(d.commands,f.func)>=0)&&d.l(f.func,f.args,e.origin)}}};
window.addEventListener("message",this.listener)}
y(ox,G);ox.prototype.sendMessage=function(a,b){if(b=b||this.i){this.sessionId&&(a.id=this.sessionId);this.channel&&(a.channel=this.channel);try{var c=JSON.stringify(a);b.postMessage(c,this.targetOrigin)}catch(d){qt(d)}}};
ox.prototype.U=function(){window.removeEventListener("message",this.listener);G.prototype.U.call(this)};var px={},qx=(px["api.invalidparam"]=2,px.auth=150,px["drm.auth"]=150,px["heartbeat.net"]=150,px["heartbeat.servererror"]=150,px["heartbeat.stop"]=150,px["html5.unsupportedads"]=5,px["fmt.noneavailable"]=5,px["fmt.decode"]=5,px["fmt.unplayable"]=5,px["html5.missingapi"]=5,px["html5.unsupportedlive"]=5,px["drm.unavailable"]=5,px["mrm.blocked"]=151,px["embedder.identity.denied"]=152,px);var rx=new Set("endSeconds startSeconds mediaContentUrl suggestedQuality videoId rct rctn".split(" "));function sx(a){return(a.search("cue")===0||a.search("load")===0)&&a!=="loadModule"}
function tx(a,b,c){if(typeof a==="string")return{videoId:a,startSeconds:b,suggestedQuality:c};b={};c=x(rx);for(var d=c.next();!d.done;d=c.next())d=d.value,a[d]&&(b[d]=a[d]);return b}
function ux(a,b,c,d){if(Ra(a)&&!Array.isArray(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};typeof a==="string"&&a.length===16?b.list="PL"+a:b.playlist=a;return b}
;function vx(){var a=wx;this.l=[];this.isReady=!1;this.v={};this.j=[];this.i=[];this.A=!1;this.m=U("web_player_split_event_bus_iframe");var b=this.h=new ox(!!T("WIDGET_ID_ENFORCE")),c=this.Ee.bind(this);b.l=c;b.commands=null;this.h.channel="widget";if(b=T("WIDGET_ID"))this.h.sessionId=b;this.api=a;xx(this,"onReady",this.onReady.bind(this));xx(this,"onVideoProgress",this.Te.bind(this));xx(this,"onVolumeChange",this.Ue.bind(this));xx(this,"onApiChange",this.Me.bind(this));xx(this,"onPlaybackQualityChange",
this.Qe.bind(this));xx(this,"onPlaybackRateChange",this.Re.bind(this));xx(this,"onStateChange",this.Se.bind(this));xx(this,"onWebglSettingsChanged",this.Ve.bind(this));xx(this,"onCaptionsTrackListChanged",this.Ne.bind(this));xx(this,"captionssettingschanged",this.Oe.bind(this))}
n=vx.prototype;
n.Ee=function(a,b,c){if(a==="addEventListener"&&b)a=b[0],a==="onReady"?this.api.logApiCall(a+" invocation",c):a==="onError"&&this.A&&(this.api.logApiCall(a+" invocation",c,this.errorCode),this.errorCode=void 0),this.api.logApiCall(a+" registration",c),this.v[a]||a==="onReady"||(b=yx(this,a,c),this.i.push({eventType:a,listener:b,origin:c}),this.m?this.api.handleExternalCall("addEventListener",[a,b],c):this.api.addEventListener(a,b),this.v[a]=!0);else if(this.api.isExternalMethodAvailable(a,c)){b=b||
[];if(b.length>0&&sx(a)){var d=b;if(Ra(d[0])&&!Array.isArray(d[0]))var e=d[0];else switch(e={},a){case "loadVideoById":case "cueVideoById":e=tx(d[0],d[1]!==void 0?Number(d[1]):void 0,d[2]);break;case "loadVideoByUrl":case "cueVideoByUrl":e=d[0];typeof e==="string"&&(e={mediaContentUrl:e,startSeconds:d[1]!==void 0?Number(d[1]):void 0,suggestedQuality:d[2]});b:{if((d=e.mediaContentUrl)&&(d=/\/([ve]|embed)\/([^#?]+)/.exec(d))&&d[2]){d=d[2];break b}d=null}e.videoId=d;e=tx(e);break;case "loadPlaylist":case "cuePlaylist":e=
ux(d[0],d[1],d[2],d[3])}b.length=1;b[0]=e}this.api.handleExternalCall(a,b,c);sx(a)&&zx(this,Ax(this))}};
n.ae=function(){this.isReady=!0;this.sendMessage("initialDelivery",Ax(this));this.sendMessage("onReady");Db(this.l,this.vd,this);this.l=[]};
function zx(a,b){a.sendMessage("infoDelivery",b)}
n.vd=function(a){this.isReady?this.h.sendMessage(a):this.l.push(a)};
n.sendMessage=function(a,b){this.vd({event:a,info:b===void 0?null:b})};
function yx(a,b,c){return function(d){b==="onError"?a.api.logApiCall(b+" invocation",c,d):a.api.logApiCall(b+" invocation",c);a.sendMessage(b,d)}}
n.onReady=function(){var a=this.h,b=this.ae.bind(this);a.h=b;a=this.api.getVideoData();if(!a.isPlayable){this.A=!0;a=a.errorCode;var c=c===void 0?5:c;this.errorCode=a?qx[a]||c:c;this.sendMessage("onError",this.errorCode.toString())}};
function xx(a,b,c){a.j.push({eventType:b,listener:c});a.api.addEventListener(b,c)}
function Ax(a){if(!a.api)return null;var b=a.api.getApiInterface();Ib(b,"getVideoData");for(var c={apiInterface:b},d=0,e=b.length;d<e;d++){var f=b[d];if(f.search("get")===0||f.search("is")===0){var g=0;f.search("get")===0?g=3:f.search("is")===0&&(g=2);g=f.charAt(g).toLowerCase()+f.substr(g+1);try{var h=a.api[f]();c[g]=h}catch(k){}}}c.videoData=a.api.getVideoData();c.currentTimeLastUpdated_=Date.now()/1E3;return c}
n.Se=function(a){a={playerState:a,currentTime:this.api.getCurrentTime(),duration:this.api.getDuration(),videoData:this.api.getVideoData(),videoStartBytes:0,videoBytesTotal:this.api.getVideoBytesTotal(),videoLoadedFraction:this.api.getVideoLoadedFraction(),playbackQuality:this.api.getPlaybackQuality(),availableQualityLevels:this.api.getAvailableQualityLevels(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getVideoUrl&&
(a.videoUrl=this.api.getVideoUrl());this.api.getVideoContentRect&&(a.videoContentRect=this.api.getVideoContentRect());this.api.getProgressState&&(a.progressState=this.api.getProgressState());this.api.getPlaylist&&(a.playlist=this.api.getPlaylist());this.api.getPlaylistIndex&&(a.playlistIndex=this.api.getPlaylistIndex());this.api.getStoryboardFormat&&(a.storyboardFormat=this.api.getStoryboardFormat());zx(this,a)};
n.Qe=function(a){a={playbackQuality:a};this.api.getAvailableQualityLevels&&(a.availableQualityLevels=this.api.getAvailableQualityLevels());this.api.getPreferredQuality&&(a.preferredQuality=this.api.getPreferredQuality());zx(this,a)};
n.Re=function(a){zx(this,{playbackRate:a})};
n.Me=function(){for(var a=this.api.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.api.getOptions(e);a.join(", ");b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var k=f[g],l=this.api.getOption(e,k);b[e][k]=l}}this.sendMessage("apiInfoDelivery",b)};
n.Ue=function(){zx(this,{muted:this.api.isMuted(),volume:this.api.getVolume()})};
n.Te=function(a){a={currentTime:a,videoBytesLoaded:this.api.getVideoBytesLoaded(),videoLoadedFraction:this.api.getVideoLoadedFraction(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getProgressState&&(a.progressState=this.api.getProgressState());zx(this,a)};
n.Ve=function(){var a={sphericalProperties:this.api.getSphericalProperties()};zx(this,a)};
n.Ne=function(){if(this.api.getCaptionTracks){var a={captionTracks:this.api.getCaptionTracks()};zx(this,a)}};
n.Oe=function(){if(this.api.getSubtitlesUserSettings){var a={subtitlesUserSettings:this.api.getSubtitlesUserSettings()};zx(this,a)}};
n.dispose=function(){this.h=null;for(var a=0;a<this.j.length;a++){var b=this.j[a];this.api.removeEventListener(b.eventType,b.listener)}this.j=[];for(a=0;a<this.i.length;a++)b=this.i[a],this.m?this.api.handleExternalCall("removeEventListener",[b.eventType,b.listener],b.origin):this.api.removeEventListener(b.eventType,b.listener);this.i=[]};function Bx(a,b){G.call(this);this.h={};this.started=!1;this.i=U("web_player_split_event_bus_iframe");this.connection=b;this.connection.subscribe("command",this.qd,this);this.api=a;this.start()}
y(Bx,G);n=Bx.prototype;n.start=function(){this.started||this.V||(this.started=!0,this.connection.jb("RECEIVING"))};
n.jb=function(a,b){this.started&&!this.V&&this.connection.jb(a,b)};
n.qd=function(a,b,c){if(this.started&&!this.V){var d=b||{};switch(a){case "addEventListener":typeof d.event==="string"&&this.addListener(d.event,c);break;case "removeEventListener":typeof d.event==="string"&&this.removeListener(d.event,c);break;default:this.api.isReady()&&this.api.isExternalMethodAvailable(a,c||null)&&(b=Cx(a,b||{}),c=this.api.handleExternalCall(a,b,c||null),(c=Dx(a,c))&&this.jb(a,c))}}};
n.addListener=function(a,b,c){a in this.h||(b=this.Pe.bind(this,a),this.h[a]=b,this.addEventListener(a,b,c))};
n.Pe=function(a,b){this.started&&!this.V&&this.connection.jb(a,Ex(a,b))};
n.removeListener=function(a,b){a in this.h&&(this.removeEventListener(a,this.h[a],b),delete this.h[a])};
n.addEventListener=function(a,b,c){this.i?this.api.handleExternalCall("addEventListener",[a,b],c||null):this.api.addEventListener(a,b)};
n.removeEventListener=function(a,b,c){this.i?this.api.handleExternalCall("removeEventListener",[a,b],c||null):this.api.removeEventListener(a,b)};
function Cx(a,b){switch(a){case "loadVideoById":return a=tx(b),[a];case "cueVideoById":return a=tx(b),[a];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return a=ux(b),[a];case "cuePlaylist":return a=ux(b),[a];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];
case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey,b.key,b.code]}return[]}
function Dx(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
function Ex(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}if(b!=null)return{value:b}}
n.U=function(){this.connection.unsubscribe("command",this.qd,this);this.connection=null;for(var a in this.h)this.h.hasOwnProperty(a)&&this.removeListener(a);G.prototype.U.call(this);delete this.api};function Fx(a,b,c){Eu.call(this);this.j=a;this.i=b;this.id=c}
y(Fx,Eu);Fx.prototype.jb=function(a,b){this.V||this.j.jb(this.i,this.id,a,b)};
Fx.prototype.U=function(){this.i=this.j=null;Eu.prototype.U.call(this)};function Gx(a,b,c){G.call(this);this.h=a;this.origin=c;this.j=wr(window,"message",this.i.bind(this));this.connection=new Fx(this,a,b);Ec(this,this.connection)}
y(Gx,G);Gx.prototype.jb=function(a,b,c,d){this.V||a!==this.h||(a={id:b,command:c},d&&(a.data=d),this.h.postMessage(JSON.stringify(a),this.origin))};
Gx.prototype.i=function(a){if(!this.V&&a.origin===this.origin){var b=a.data;if(typeof b==="string"){try{b=JSON.parse(b)}catch(d){return}if(b.command){var c=this.connection;c.V||c.l("command",b.command,b.data,a.origin)}}}};
Gx.prototype.U=function(){yr(this.j);this.h=null;G.prototype.U.call(this)};var Hx=new bx;function Ix(){return Hx.Qc()}
function Jx(a){a=a===void 0?{}:a;return Hx.invoke(a)}
;function Kx(a,b,c,d,e){G.call(this);var f=this;this.A=b;this.webPlayerContextConfig=d;this.uc=e;this.Za=!1;this.api={};this.ia=this.m=null;this.W=new M;this.h={};this.da=this.xa=this.elementId=this.Cb=this.config=null;this.ba=!1;this.j=this.H=null;this.Ga={};this.vc=["onReady"];this.lastError=null;this.Tb=NaN;this.P={};this.ga=0;this.i=this.l=a;Ec(this,this.W);Lx(this);c?this.ga=setTimeout(function(){f.loadNewVideoConfig(c)},0):d&&(Mx(this),Nx(this))}
y(Kx,G);n=Kx.prototype;n.getId=function(){return this.A};
n.loadNewVideoConfig=function(a){if(!this.V){this.ga&&(clearTimeout(this.ga),this.ga=0);var b=a||{};b instanceof gu||(b=new gu(b));this.config=b;this.setConfig(a);Nx(this);this.isReady()&&Ox(this)}};
function Mx(a){var b;a.webPlayerContextConfig?b=a.webPlayerContextConfig.rootElementId:b=a.config.attrs.id;a.elementId=b||a.elementId;a.elementId==="video-player"&&(a.elementId=a.A,a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.A:a.config.attrs.id=a.A);var c;((c=a.i)==null?void 0:c.id)===a.elementId&&(a.elementId+="-player",a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.elementId:a.config.attrs.id=a.elementId)}
n.setConfig=function(a){this.Cb=a;this.config=Px(a);Mx(this);if(!this.xa){var b;this.xa=Qx(this,((b=this.config.args)==null?void 0:b.jsapicallback)||"onYouTubePlayerReady")}this.config.args?this.config.args.jsapicallback=null:this.config.args={jsapicallback:null};var c;if((c=this.config)==null?0:c.attrs)a=this.config.attrs,(b=a.width)&&this.i&&(this.i.style.width=Vi(Number(b)||b)),(a=a.height)&&this.i&&(this.i.style.height=Vi(Number(a)||a))};
function Ox(a){if(a.config&&a.config.loaded!==!0)if(a.config.loaded=!0,!a.config.args||a.config.args.autoplay!=="0"&&a.config.args.autoplay!==0&&a.config.args.autoplay!==!1){var b;a.api.loadVideoByPlayerVars((b=a.config.args)!=null?b:null)}else a.api.cueVideoByPlayerVars(a.config.args)}
function Rx(a){var b=!0,c=Sx(a);c&&a.config&&(b=c.dataset.version===Tx(a));return b&&!!E("yt.player.Application.create")}
function Nx(a){if(!a.V&&!a.ba){var b=Rx(a);if(b&&(Sx(a)?"html5":null)==="html5")a.da="html5",a.isReady()||Ux(a);else if(Vx(a),a.da="html5",b&&a.j&&a.l)a.l.appendChild(a.j),Ux(a);else{a.config&&(a.config.loaded=!0);var c=!1;a.H=function(){c=!0;var d=Wx(a,"player_bootstrap_method")?E("yt.player.Application.createAlternate")||E("yt.player.Application.create"):E("yt.player.Application.create");var e=a.config?Px(a.config):void 0;d&&d(a.l,e,a.webPlayerContextConfig,a.uc);Ux(a)};
a.ba=!0;b?a.H():(pu(Tx(a),a.H),(b=Xx(a))&&wu(b||""),Yx(a)&&!c&&D("yt.player.Application.create",null))}}}
function Sx(a){var b=Fd(a.elementId);!b&&a.i&&a.i.querySelector&&(b=a.i.querySelector("#"+a.elementId));return b}
function Ux(a){if(!a.V){var b=Sx(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);if(c){a.ba=!1;if(!Wx(a,"html5_remove_not_servable_check_killswitch")){var d;if((b==null?0:b.isNotServable)&&a.config&&(b==null?0:b.isNotServable((d=a.config.args)==null?void 0:d.video_id)))return}Zx(a)}else a.Tb=setTimeout(function(){Ux(a)},50)}}
function Zx(a){Lx(a);a.Za=!0;var b=Sx(a);if(b){a.m=$x(a,b,"addEventListener");a.ia=$x(a,b,"removeEventListener");var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=a.api,e=0;e<c.length;e++){var f=c[e];d[f]||(d[f]=$x(a,b,f))}}for(var g in a.h)a.h.hasOwnProperty(g)&&a.m&&a.m(g,a.h[g]);Ox(a);a.xa&&a.xa(a.api);a.W.Ya("onReady",a.api)}
function $x(a,b,c){var d=b[c];return function(){var e=B.apply(0,arguments);try{return a.lastError=null,d.apply(b,e)}catch(f){if(c!=="sendAbandonmentPing")throw f.params=c,a.lastError=f,e=new V("PlayerProxy error in method call",{error:f,method:c,playerId:a.A}),e.level="WARNING",e;}}}
function Lx(a){a.Za=!1;if(a.ia)for(var b in a.h)a.h.hasOwnProperty(b)&&a.ia(b,a.h[b]);for(var c in a.P)a.P.hasOwnProperty(c)&&clearTimeout(Number(c));a.P={};a.m=null;a.ia=null;b=a.api;for(var d in b)b.hasOwnProperty(d)&&(b[d]=null);b.addEventListener=function(e,f){a.addEventListener(e,f)};
b.removeEventListener=function(e,f){a.removeEventListener(e,f)};
b.destroy=function(){a.dispose()};
b.getLastError=function(){return a.getLastError()};
b.getPlayerType=function(){return a.getPlayerType()};
b.getCurrentVideoConfig=function(){return a.Cb};
b.loadNewVideoConfig=function(e){a.loadNewVideoConfig(e)};
b.isReady=function(){return a.isReady()}}
n.isReady=function(){return this.Za};
n.addEventListener=function(a,b){var c=this,d=Qx(this,b);d&&(Cb(this.vc,a)>=0||this.h[a]||(b=ay(this,a),this.m&&this.m(a,b)),this.W.subscribe(a,d),a==="onReady"&&this.isReady()&&setTimeout(function(){d(c.api)},0))};
n.removeEventListener=function(a,b){this.V||(b=Qx(this,b))&&this.W.unsubscribe(a,b)};
function Qx(a,b){var c=b;if(typeof b==="string"){if(a.Ga[b])return a.Ga[b];c=function(){var d=B.apply(0,arguments),e=E(b);if(e)try{e.apply(C,d)}catch(f){throw d=new V("PlayerProxy error when executing callback",{error:f}),d.level="ERROR",d;}};
a.Ga[b]=c}return c?c:null}
function ay(a,b){function c(d){var e=setTimeout(function(){if(!a.V){try{a.W.Ya(b,d!=null?d:void 0)}catch(h){var f=new V("PlayerProxy error when creating global callback",{error:h.message,event:b,playerId:a.A,data:d,originalStack:h.stack});f.level="WARNING";throw f;}f=a.P;var g=String(e);g in f&&delete f[g]}},0);
Ob(a.P,String(e))}
return a.h[b]=c}
n.getPlayerType=function(){return this.da||(Sx(this)?"html5":null)};
n.getLastError=function(){return this.lastError};
function Vx(a){a.cancel();Lx(a);a.da=null;a.config&&(a.config.loaded=!1);var b=Sx(a);b&&(Rx(a)||!Yx(a)?a.j=b:(b&&b.destroy&&b.destroy(),a.j=null));if(a.l)for(a=a.l;b=a.firstChild;)a.removeChild(b)}
n.cancel=function(){this.H&&tu(Tx(this),this.H);clearTimeout(this.Tb);this.ba=!1};
n.U=function(){Vx(this);if(this.j&&this.config&&this.j.destroy)try{this.j.destroy()}catch(b){var a=new V("PlayerProxy error during disposal",{error:b});a.level="ERROR";throw a;}this.Ga=null;for(a in this.h)this.h.hasOwnProperty(a)&&delete this.h[a];this.Cb=this.config=this.api=null;delete this.l;delete this.i;G.prototype.U.call(this)};
function Yx(a){var b,c;a=(b=a.config)==null?void 0:(c=b.args)==null?void 0:c.fflags;return!!a&&a.indexOf("player_destroy_old_version=true")!==-1}
function Tx(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.jsUrl:(a=a.config.assets)?a.js:""}
function Xx(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.cssUrl:(a=a.config.assets)?a.css:""}
function Wx(a,b){if(a.webPlayerContextConfig)var c=a.webPlayerContextConfig.serializedExperimentFlags;else{var d;if((d=a.config)==null?0:d.args)c=a.config.args.fflags}return(c||"").split("&").includes(b+"=true")}
function Px(a){for(var b={},c=x(Object.keys(a)),d=c.next();!d.done;d=c.next()){d=d.value;var e=a[d];b[d]=typeof e==="object"?Rb(e):e}return b}
;var by={},cy="player_uid_"+(Math.random()*1E9>>>0);function dy(a,b){var c="player",d=!1;d=d===void 0?!0:d;c=typeof c==="string"?Fd(c):c;var e=cy+"_"+Sa(c),f=by[e];if(f&&d)return ey(a,b)?f.api.loadVideoByPlayerVars(a.args||null):f.loadNewVideoConfig(a),f.api;f=new Kx(c,e,a,b,void 0);by[e]=f;f.addOnDisposeCallback(function(){delete by[f.getId()]});
return f.api}
function ey(a,b){return b&&b.serializedExperimentFlags?b.serializedExperimentFlags.includes("web_player_remove_playerproxy=true"):a&&a.args&&a.args.fflags?a.args.fflags.includes("web_player_remove_playerproxy=true"):!1}
;var wx=null,fy=null,gy=null;
function hy(){iw();var a=Fm(),b=Im(119),c=window.devicePixelRatio>1;if(document.body&&kj(document.body,"exp-invert-logo"))if(c&&!kj(document.body,"inverted-hdpi")){var d=document.body;if(d.classList)d.classList.add("inverted-hdpi");else if(!kj(d,"inverted-hdpi")){var e=ij(d);jj(d,e+(e.length>0?" inverted-hdpi":"inverted-hdpi"))}}else!c&&kj(document.body,"inverted-hdpi")&&lj();if(b!=c){b="f"+(Math.floor(119/31)+1);d=Jm(b)||0;d=c?d|67108864:d&-67108865;d===0?delete Cm[b]:(c=d.toString(16),Cm[b]=c.toString());
c=!0;U("web_secure_pref_cookie_killswitch")&&(c=!1);b=a.h;d=[];for(f in Cm)Cm.hasOwnProperty(f)&&d.push(f+"="+encodeURIComponent(String(Cm[f])));var f=d.join("&");ym(b,f,63072E3,a.i,c)}}
function iy(){jy()}
function ky(){hw("ep_init_pr");jy()}
function jy(){var a=wx.getVideoData(1);a=a.title?a.title+" - YouTube":"YouTube";document.title!==a&&(document.title=a)}
function ly(){wx&&wx.sendAbandonmentPing&&wx.sendAbandonmentPing();T("PL_ATT")&&Hx.dispose();for(var a=cj,b=0,c=Xw.length;b<c;b++)a.qa(Xw[b]);Xw.length=0;ru("//static.doubleclick.net/instream/ad_status.js");Yw=!1;pl("DCLKSTAT",0);Dc(gy,fy);wx&&(wx.removeEventListener("onVideoDataChange",iy),wx.destroy())}
;D("yt.setConfig",pl);D("yt.config.set",pl);D("yt.setMsg",ou);D("yt.msgs.set",ou);D("yt.logging.errors.log",pt);
D("writeEmbed",function(){var a=T("PLAYER_CONFIG");if(!a){var b=T("PLAYER_VARS");b&&(a={args:b})}av(!0);a.args.ps==="gvn"&&(document.body.style.backgroundColor="transparent");a.attrs||(a.attrs={width:"100%",height:"100%",id:"video-player"});var c=document.referrer;b=T("POST_MESSAGE_ORIGIN");window!==window.top&&c&&c!==document.URL&&(a.args.loaderUrl=c);dw("embed",["ol"]);c=Hw();if(!c.serializedForcedExperimentIds){var d=Dl(window.location.href);d.forced_experiments&&(c.serializedForcedExperimentIds=
d.forced_experiments)}var e;((e=a.args)==null?0:e.autoplay)&&dw("watch",["pbs","pbu","pbp"]);wx=dy(a,c);wx.addEventListener("onVideoDataChange",iy);wx.addEventListener("onReady",ky);a=T("POST_MESSAGE_ID","player");T("ENABLE_JS_API")?gy=new vx:T("ENABLE_POST_API")&&typeof a==="string"&&typeof b==="string"&&(fy=new Gx(window.parent,a,b),gy=new Bx(wx,fy.connection));Zw();U("ytidb_create_logger_embed_killswitch")||En();a={};hx.h||(hx.h=new hx);hx.h.install((a.flush_logs={callback:function(){Vs()}},a));
nr();U("ytidb_clear_embedded_player")&&cj.pa(function(){var f,g;if(!Fw){var h=$r();Wr(h,{oc:Ew,Bd:Cw});var k={Xc:{feedbackEndpoint:nv(ww),modifyChannelNotificationPreferenceEndpoint:nv(xw),playlistEditEndpoint:nv(yw),subscribeEndpoint:nv(uw),unsubscribeEndpoint:nv(vw),webPlayerShareEntityServiceEndpoint:nv(zw)}},l=jv(),m={};l&&(m.client_location=l);f===void 0&&(f=um());g===void 0&&(g=h.resolve(Ew));nw(k,g,f,m);Wr(h,{oc:tw,Cd:mw.h});Fw=h.resolve(tw)}Tw()})});
D("yt.abuse.player.botguardInitialized",E("yt.abuse.player.botguardInitialized")||Ix);D("yt.abuse.player.invokeBotguard",E("yt.abuse.player.invokeBotguard")||Jx);D("yt.abuse.dclkstatus.checkDclkStatus",E("yt.abuse.dclkstatus.checkDclkStatus")||$w);D("yt.player.exports.navigate",E("yt.player.exports.navigate")||$u);D("yt.util.activity.init",E("yt.util.activity.init")||Er);D("yt.util.activity.getTimeSinceActive",E("yt.util.activity.getTimeSinceActive")||Hr);
D("yt.util.activity.setTimestamp",E("yt.util.activity.setTimestamp")||Fr);window.addEventListener("load",tl(function(){hy()}));
window.addEventListener("pageshow",tl(function(a){a.persisted||hy()}));
window.addEventListener("pagehide",tl(function(a){U("embeds_web_enable_dispose_player_if_page_not_cached_killswitch")?ly():a.persisted||ly()}));
window.onerror=function(a,b,c,d,e){b=b===void 0?"Unknown file":b;c=c===void 0?0:c;var f=!1,g=ql("log_window_onerror_fraction");if(g&&Math.random()<g)f=!0;else{g=document.getElementsByTagName("script");for(var h=0,k=g.length;h<k;h++)if(g[h].src.indexOf("/debug-")>0){f=!0;break}}f&&(f=!1,e?f=!0:(typeof a==="string"?g=a:ErrorEvent&&a instanceof ErrorEvent?(f=!0,g=a.message,b=a.filename,c=a.lineno,d=a.colno):(g="Unknown error",b="Unknown file",c=0),e=new V(g),e.name="UnhandledWindowError",e.message=g,
e.fileName=b,e.lineNumber=c,isNaN(d)?delete e.columnNumber:e.columnNumber=d),f?pt(e):qt(e))};
je=rt;window.addEventListener("unhandledrejection",function(a){rt(a.reason)});
Db(T("ERRORS")||[],function(a){pt.apply(null,a)});
pl("ERRORS",[]);}).call(this);
