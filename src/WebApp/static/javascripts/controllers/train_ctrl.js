cbirCtrls.controller('TrainingCtrl', ['$scope', '$http', '$timeout',  function($scope, $http, $timeout) {

  $scope.parameters = {
    dataset: null,
    batch_size: 200,
    max_epoch: 5,
    learning_rate: 0.1,
    momentum_rate: 0,
    weight_decay: 0,
    lambda_l1: 0
  }

  var timeout_promise

  $http.get('/get_datasets_name').
        success(function(data) {
            $scope.datasets = data;
        });

  $scope.submit = function() {
    $http({
        method  : 'POST',
        url     : 'train_network',
        data    : $.param($scope.parameters),  // pass in data as strings
        headers : { 'Content-Type': 'application/x-www-form-urlencoded' }  // set the headers so angular passing info as form data (not request payload)
        })
        .success(function(data) {

            console.log(data)
            $scope.task_id = data.taskid

            var error_rates = {"start":1,"end":1,"step":1,"names":["Train errors","Validation errors"],"values": [[],[]]}
            error_rates["displayNames"] = ["Train errors","Validation errors"];
            error_rates["colors"] = ["red", "green"];

            var l1 = new LineGraph({containerId: 'graph1', data: error_rates});

            $scope.get_error_rates = function() {

                $http({
                  url:'/task_result',
                  method: "GET",
                  params: {'taskid': $scope.task_id}
                }).success(function(data) {

                  if (data.status == 'PENDING') {

                    $http.get('/get_error_rates').
                      success(function(data) {

                        length = data[1].length
                        console.log(length)
                        if (length == 1) {
                          error_rates.end = data[1].length
                          error_rates.values = [data[1], data[2]]
                        }
                        else if (length > 1 && length !=  error_rates.end) {

                          error_rates.end = data[1].length
                          error_rates.values = [data[1], data[2]]
                        }

                        console.log(error_rates)
                        l1.updateData(error_rates);

                      });
                  }
                  else if (data.status == 'SUCCESS') {

                    $timeout.cancel(timeout_promise)
                  }
                });
           };

           $scope.intervalFunction = function(){
              timeout_promise = $timeout(function() {
                               $scope.get_error_rates();
                               $scope.intervalFunction();
                             }, 5000)
           };

           $scope.intervalFunction()

        });
      };
}]);
