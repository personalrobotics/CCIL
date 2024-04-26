window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    bulmaSlider.attach();

    $("#accordion").accordion({
      collapsible: true,
      active: false,
      heightStyle: "content",
      beforeActivate: function(event, ui) {
        if (ui.newPanel.text() !== "") {
          $("#accordion .collapsed-text").addClass("display-none");
          $("#accordion .collapsed-icon").addClass("display-none");
          $("#accordion .expanded-text").removeClass("display-none");
          $("#accordion .expanded-icon").removeClass("display-none");
        } else {
          $("#accordion .expanded-text").addClass("display-none");
          $("#accordion .expanded-icon").addClass("display-none");
          $("#accordion .collapsed-text").removeClass("display-none");
          $("#accordion .collapsed-icon").removeClass("display-none");
        }
      }
    });
    // unhide content that was collapsed into accordion
    $("#accordion > div.display-none").removeClass("display-none");
})
