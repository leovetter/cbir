var CbirApp = angular
                .module('CBIR', ['ngFileUpload', 'ngCookies', 'ngRoute', 'CbirCtrls',
                                 'CbirDirectives', 'angular.filter', 'n3-line-chart']);


CbirApp.config(['$interpolateProvider', '$routeProvider', '$httpProvider',
                function($interpolateProvider, $routeProvider, $httpProvider) {

  $httpProvider.defaults.xsrfCookieName = 'csrftoken';
  $httpProvider.defaults.xsrfHeaderName = 'X-CSRFToken';
  $httpProvider.defaults.withCredentials = true;

  $interpolateProvider.startSymbol('{[{');
  $interpolateProvider.endSymbol('}]}');

  $routeProvider.
      when('/search', {
        templateUrl: 'static/partials/search.html',
        controller: 'SearchCtrl'
      }).
      when('/training', {
        templateUrl: 'static/partials/training.html',
        controller: 'TrainingCtrl'
      }).
      when('/evaluation', {
        templateUrl: 'static/partials/evaluation.html',
        controller: 'EvaluationCtrl'
      }).
      otherwise({
        redirectTo: '/search'
      });
}]);
