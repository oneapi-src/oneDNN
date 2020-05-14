//configure tms    
var wapLocalCode = 'us-en'; 
//dynamically set per localized site, see mapping table for values    
var wapSection = "oneapi";     
//load tms    
(function() {                 
	var host = (window.document.location.protocol == 'http:') ? "http://www.intel.com" : "https://www.intel.com";
	var url = host+"/content/dam/www/global/wap/tms-loader.js"; //wap file url
	var po = document.createElement('script'); 
	po.type = 'text/javascript'; 
	po.async = true;  
	po.src = url;                 
	var s = document.getElementsByTagName('script')[0]; 
	s.parentNode.insertBefore(po, s);     
})(); 
