'use strict';

function stickyNav(nav, navOffset) {
    window.onscroll = function() {        
        const elemAfterNav = document.querySelector('div#navrow1 + *') ||
            document.querySelector('.header');
        const searchResults = document.querySelector('#MSearchResultsWindow');
        if(window.pageYOffset >= navOffset) {
            elemAfterNav.style.paddingTop = '36px';
            nav.classList.add('sticky');
        }
        else {
            searchResults.style.top = '150px';
            elemAfterNav.style.paddingTop = '0px';
            nav.classList.remove('sticky');
        }
    };

}

window.addEventListener(
    'load',
    function init() {
        const nav = document.querySelector('#navrow1');
        const navOffset = nav.offsetTop;
        stickyNav(nav, navOffset);
    },
    false
);