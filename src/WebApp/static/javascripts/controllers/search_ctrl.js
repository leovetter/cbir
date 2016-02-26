cbirCtrls.controller('SearchCtrl', ['$scope', '$http', 'Upload', '$cookies',
                                    function($scope, $http, Upload, $cookies) {

  $scope.upImages = [];
  $scope.images = [];
  $scope.algos = ['ConvNet', 'BOW'];

  $scope.$watch('file', function(newValue, oldValue) {

      if(newValue) {
        $scope.upImages = []
        $scope.upImages.push(newValue)
        $('#search_input').attr('placeholder', newValue.name)
      }
  });

  $scope.upload = function (file, algo) {

    console.log($(this))
    Upload.upload({
                    url: 'http://localhost:8000/image',
                    headers: {
                      'X-CSRFToken': $cookies.get('csrftoken')
                    },
                    data: {file: file, 'algo': algo}
                }).then(function (resp) {
                    $scope.images = resp.data
                    console.log('Success ' + resp.config.data.file.name + 'uploaded. Response: ' + resp.data);
                }, function (resp) {
                    // console.log('Error status: ' + resp.status);
                }, function (evt) {
                    // var progressPercentage = parseInt(100.0 * evt.loaded / evt.total);
                    // console.log('progress: ' + progressPercentage + '% ' + evt.config.data.file.name);
                });
    };

  $scope.submit = function(algo) {

    if ($scope.file) {
      $scope.upload($scope.file, algo);
    }
  };

}])
