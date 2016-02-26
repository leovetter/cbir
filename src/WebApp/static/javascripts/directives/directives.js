'use strict';



var cbirDirectives = angular.module('CbirDirectives', []);

    /**
    * The ng-thumb directive
    * @author: nerv
    * @version: 0.1.2, 2014-01-09
    */
cbirDirectives.directive('preview', ['$window', function ($window) {

        return {
            restrict: 'A',
            template: '<canvas/>',
            link: function (scope, element, attributes) {

                function onLoadFile(event) {

                    var img = new Image();
                    img.onload = onLoadImage;
                    img.src = event.target.result;

                }
                function onLoadImage() {
                  var width = this.width * 3
                  var height = this.height * 3
                  canvas.attr({width: width, height: height});
                  canvas[0].getContext('2d').drawImage(this, 0, 0, width, height);

                  $('#img-preview').affix({
                        offset: {
                          top: 200
                        }
                  });
                }
                var params = scope.$eval(attributes.preview);
                var canvas = element.find('canvas');
                var reader = new FileReader();
                reader.onload = onLoadFile;
                reader.readAsDataURL(params.file);
            }
        };
    }]);
